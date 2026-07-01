//! Core image type and I/O.

use crate::header::{Nifti1Header, NiftiError, read_file_bytes, dtype,
    OFF_DIM, OFF_DATATYPE, OFF_BITPIX, OFF_PIXDIM, OFF_VOX_OFFSET,
    OFF_SROW_X, OFF_SROW_Y, OFF_SROW_Z, OFF_SFORM_CODE, OFF_QFORM_CODE,
    OFF_QUATERN_B, OFF_QUATERN_C, OFF_QUATERN_D,
    OFF_QOFFSET_X, OFF_QOFFSET_Y, OFF_QOFFSET_Z, OFF_MAGIC};
use bytemuck::Pod;
use ndarray::prelude::*;
use ndarray::Array2;
use rayon::prelude::*;
use std::fmt;
use std::path::Path;

// ─── Datatype trait ─────────────────────────────────────────────────────────

/// Trait mapping Rust numeric types to NIfTI-1 datatype codes.
///
/// Users don't need to implement this; it's automatically implemented
/// for all standard NIfTI-compatible types.
pub trait NiftiType: Pod + Send + Sync + 'static {
    const DATATYPE: i16;
    const BITPIX: i16;
}

impl NiftiType for f32   { const DATATYPE: i16 = dtype::FLOAT32;  const BITPIX: i16 = 32; }
impl NiftiType for f64   { const DATATYPE: i16 = dtype::FLOAT64;  const BITPIX: i16 = 64; }
impl NiftiType for u8    { const DATATYPE: i16 = dtype::UINT8;    const BITPIX: i16 = 8;  }
impl NiftiType for i8    { const DATATYPE: i16 = dtype::INT8;     const BITPIX: i16 = 8;  }
impl NiftiType for u16   { const DATATYPE: i16 = dtype::UINT16;   const BITPIX: i16 = 16; }
impl NiftiType for i16   { const DATATYPE: i16 = dtype::INT16;    const BITPIX: i16 = 16; }
impl NiftiType for u32   { const DATATYPE: i16 = dtype::UINT32;   const BITPIX: i16 = 32; }
impl NiftiType for i32   { const DATATYPE: i16 = dtype::INT32;    const BITPIX: i16 = 32; }
impl NiftiType for i64   { const DATATYPE: i16 = dtype::INT64;    const BITPIX: i16 = 64; }
impl NiftiType for u64   { const DATATYPE: i16 = dtype::UINT64;   const BITPIX: i16 = 64; }

// ─── 3×3 matrix inverse (ndarray only, no nalgebra) ─────────────────────────

/// Compute the inverse of a 3×3 matrix using the analytical formula.
pub(crate) fn mat3_inv(m: ArrayView2<f64>) -> Array2<f64> {
    debug_assert_eq!(m.shape(), &[3, 3]);
    let a = m[[0, 0]]; let b = m[[0, 1]]; let c = m[[0, 2]];
    let d = m[[1, 0]]; let e = m[[1, 1]]; let f = m[[1, 2]];
    let g = m[[2, 0]]; let h = m[[2, 1]]; let i = m[[2, 2]];

    let det = a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g);

    assert!(det.abs() > 1e-30, "Singular 3×3 matrix in affine");

    let inv_det = 1.0 / det;
    Array2::from_shape_vec((3, 3), vec![
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ]).unwrap()
}

/// Invert a 4×4 affine matrix [[R, t], [0, 1]] using the 3×3 inverse.
fn affine_inv(aff: ArrayView2<f64>) -> Array2<f64> {
    debug_assert_eq!(aff.shape(), &[4, 4]);

    let r = aff.slice(s![..3, ..3]);
    let t = aff.slice(s![..3, 3]);

    let r_inv = mat3_inv(r);

    // t_inv = -R⁻¹ · t
    let t_inv = r_inv.dot(&t);

    let mut result = Array2::zeros((4, 4));
    result.slice_mut(s![..3, ..3]).assign(&r_inv);
    result.slice_mut(s![..3, 3]).assign(&t_inv);
    result[[3, 3]] = 1.0;
    result
}

// ─── Nifti1Image ────────────────────────────────────────────────────────────

/// Core struct: a NIfTI-1 image = header + ndarray voxel data.
#[derive(Clone)]
pub struct Nifti1Image<T> {
    pub header: Nifti1Header,
    pub ndarray: Array3<T>,
}

impl<T> Nifti1Image<T>
where
    T: NiftiType,
{
    // ─── Read ───────────────────────────────────────────────────────────────

    /// Read a NIfTI-1 image from disk (.nii or .nii.gz).
    pub fn read(path: impl AsRef<Path>) -> Result<Self, NiftiError> {
        let path = path.as_ref();
        let bytes = read_file_bytes(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse from raw file bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, NiftiError> {
        // Minimum size: 348 byte header + some data
        if bytes.len() < 348 {
            return Err(NiftiError::InvalidHeader("file too small"));
        }

        let header = Nifti1Header::parse(bytes)?;

        // Validate datatype
        if header.datatype != T::DATATYPE {
            return Err(NiftiError::DataTypeMismatch {
                expected: T::DATATYPE,
                found: header.datatype,
            });
        }

        // Get shape from header (in nifti-rs = [x, y, z] order)
        let shape = header.shape();
        if shape.len() < 3 {
            return Err(NiftiError::DimensionMismatch(
                "expected at least 3 dimensions",
            ));
        }
        let nx = shape[0];
        let ny = shape[1];
        let nz = shape[2];

        // Data start offset
        let data_off = header.vox_offset as usize;

        // Expected number of elements
        let n_expected = nx * ny * nz;
        let dtype_size = (header.bitpix / 8) as usize;
        let n_bytes_available = bytes.len().saturating_sub(data_off);
        let n_available = n_bytes_available / dtype_size;

        if n_available < n_expected {
            return Err(NiftiError::InvalidHeader(
                "file too short for declared dimensions",
            ));
        }

        // Cast raw bytes to &[T] (safe via bytemuck when T: Pod)
        let raw_slice = &bytes[data_off..data_off + n_expected * dtype_size];
        let data_slice: &[T] = bytemuck::cast_slice(raw_slice);

        // [x, y, z] (file) → [z, y, x] (ITK style, contiguous)
        // File stores data in [x,y,z] order: idx = x + nx*y + nx*ny*z
        // We reorganize so that arr_zyx[z,y,x] = file_data[x + nx*y + nx*ny*z]
        let mut itk_data = Vec::with_capacity(nx * ny * nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + nx * y + nx * ny * z;
                    itk_data.push(data_slice[idx]);
                }
            }
        }
        let ndarray = Array3::from_shape_vec((nz, ny, nx), itk_data)
            .map_err(|_| NiftiError::DimensionMismatch("shape mismatch"))?;

        Ok(Nifti1Image { header, ndarray })
    }

    // ─── Write ──────────────────────────────────────────────────────────────

    /// Write image to disk (.nii or .nii.gz based on extension).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), NiftiError> {
        let path = path.as_ref();
        let is_gz = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gz"))
            .unwrap_or(false);

        let bytes = self.to_bytes()?;

        if is_gz {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            use std::io::Write;
            encoder.write_all(&bytes)?;
            let compressed = encoder.finish()?;
            std::fs::write(path, compressed)?;
        } else {
            std::fs::write(path, bytes)?;
        }

        Ok(())
    }

    /// Serialize to NIfTI-1 bytes (header + data).
    fn to_bytes(&self) -> Result<Vec<u8>, NiftiError> {
        let hdr = &self.header;
        let shape = self.ndarray.shape(); // [z, y, x] ITK
        let nz = shape[0];
        let ny = shape[1];
        let nx = shape[2];

        // [z, y, x] (ITK) → [x, y, z] (file storage)
        let n = nx * ny * nz;
        let mut file_data: Vec<T> = Vec::with_capacity(n);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    file_data.push(self.ndarray[[z, y, x]]);
                }
            }
        }
        let raw_data: &[u8] = bytemuck::cast_slice(&file_data);

        let vox_offset = 352.0; // 348 + 4 byte "extension" dummy
        let total_size = vox_offset as usize + raw_data.len();
        let mut buf = vec![0u8; total_size];

        // ── sizeof_hdr ──
        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        // ── dim ──
        for i in 0..8 {
            let val = hdr.dim[i] as i16;
            buf[OFF_DIM + i * 2..OFF_DIM + i * 2 + 2].copy_from_slice(&val.to_le_bytes());
        }

        // ── datatype / bitpix ──
        buf[OFF_DATATYPE..OFF_DATATYPE + 2].copy_from_slice(&T::DATATYPE.to_le_bytes());
        buf[OFF_BITPIX..OFF_BITPIX + 2].copy_from_slice(&T::BITPIX.to_le_bytes());

        // ── pixdim ──
        for i in 0..8 {
            let val = hdr.pixdim[i] as f32;
            buf[OFF_PIXDIM + i * 4..OFF_PIXDIM + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        // ── vox_offset ──
        buf[OFF_VOX_OFFSET..OFF_VOX_OFFSET + 4].copy_from_slice(&(vox_offset as f32).to_le_bytes());

        // ── srow_x/y/z ──
        for i in 0..4 {
            buf[OFF_SROW_X + i * 4..OFF_SROW_X + i * 4 + 4]
                .copy_from_slice(&(hdr.srow_x[i] as f32).to_le_bytes());
            buf[OFF_SROW_Y + i * 4..OFF_SROW_Y + i * 4 + 4]
                .copy_from_slice(&(hdr.srow_y[i] as f32).to_le_bytes());
            buf[OFF_SROW_Z + i * 4..OFF_SROW_Z + i * 4 + 4]
                .copy_from_slice(&(hdr.srow_z[i] as f32).to_le_bytes());
        }

        // ── sform_code ──
        buf[OFF_SFORM_CODE..OFF_SFORM_CODE + 2].copy_from_slice(&(hdr.sform_code as i16).to_le_bytes());
        // ── qform_code ──
        buf[OFF_QFORM_CODE..OFF_QFORM_CODE + 2].copy_from_slice(&(hdr.qform_code as i16).to_le_bytes());

        // ── quaternions ──
        buf[OFF_QUATERN_B..OFF_QUATERN_B + 4].copy_from_slice(&(hdr.quatern_b as f32).to_le_bytes());
        buf[OFF_QUATERN_C..OFF_QUATERN_C + 4].copy_from_slice(&(hdr.quatern_c as f32).to_le_bytes());
        buf[OFF_QUATERN_D..OFF_QUATERN_D + 4].copy_from_slice(&(hdr.quatern_d as f32).to_le_bytes());

        buf[OFF_QOFFSET_X..OFF_QOFFSET_X + 4].copy_from_slice(&(hdr.qoffset_x as f32).to_le_bytes());
        buf[OFF_QOFFSET_Y..OFF_QOFFSET_Y + 4].copy_from_slice(&(hdr.qoffset_y as f32).to_le_bytes());
        buf[OFF_QOFFSET_Z..OFF_QOFFSET_Z + 4].copy_from_slice(&(hdr.qoffset_z as f32).to_le_bytes());

        // ── magic ──
        buf[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(b"n+1\0");

        // ── data ──
        buf[vox_offset as usize..].copy_from_slice(raw_data);

        Ok(buf)
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    /// Reference to the header.
    pub fn header(&self) -> &Nifti1Header {
        &self.header
    }

    /// Mutable reference to the header.
    pub fn header_mut(&mut self) -> &mut Nifti1Header {
        &mut self.header
    }

    /// Reference to the voxel data array (ITK style: [z, y, x]).
    pub fn ndarray(&self) -> &Array3<T> {
        &self.ndarray
    }

    /// Consume and return the array (ITK style: [z, y, x]).
    pub fn into_ndarray(self) -> Array3<T> {
        self.ndarray
    }

    /// Image size in voxels (ITK style: [x, y, z]).
    pub fn get_size(&self) -> [u32; 3] {
        let s = self.ndarray.shape();
        [s[2] as u32, s[1] as u32, s[0] as u32]
    }

    /// Voxel spacing (ITK style: [x, y, z]).
    pub fn get_spacing(&self) -> [f64; 3] {
        [
            self.header.pixdim[1],
            self.header.pixdim[2],
            self.header.pixdim[3],
        ]
    }

    /// Voxel origin in LPS (ITK style: [x, y, z]).
    ///
    /// Derived from the affine's translation column with RAS→LPS conversion.
    pub fn get_origin(&self) -> [f64; 3] {
        let aff = self.get_affine();
        [
            -aff[[0, 3]],
            -aff[[1, 3]],
            aff[[2, 3]],
        ]
    }

    /// Direction cosines in LPS (ITK style, 3×3).
    pub fn get_direction(&self) -> [[f64; 3]; 3] {
        let aff = self.get_affine();
        let a = aff.slice(s![..3, ..3]);

        // Extract spacing per column
        let sx = (a[[0, 0]].powi(2) + a[[1, 0]].powi(2) + a[[2, 0]].powi(2)).sqrt();
        let sy = (a[[0, 1]].powi(2) + a[[1, 1]].powi(2) + a[[2, 1]].powi(2)).sqrt();
        let sz = (a[[0, 2]].powi(2) + a[[1, 2]].powi(2) + a[[2, 2]].powi(2)).sqrt();

        // Normalise columns → direction cosines in RAS
        let d_ras = [
            [a[[0, 0]] / sx, a[[0, 1]] / sy, a[[0, 2]] / sz],
            [a[[1, 0]] / sx, a[[1, 1]] / sy, a[[1, 2]] / sz],
            [a[[2, 0]] / sx, a[[2, 1]] / sy, a[[2, 2]] / sz],
        ];

        // RAS → LPS (negate first two rows)
        [
            [-d_ras[0][0], -d_ras[0][1], -d_ras[0][2]],
            [-d_ras[1][0], -d_ras[1][1], -d_ras[1][2]],
            [ d_ras[2][0],  d_ras[2][1],  d_ras[2][2]],
        ]
    }

    /// Unit voxel volume in mm³.
    pub fn get_unit_size(&self) -> f64 {
        let s = self.get_spacing();
        s[0] * s[1] * s[2]
    }

    /// 4×4 affine matrix in nibabel convention (rows = axes).
    pub fn get_affine(&self) -> Array2<f64> {
        self.header.affine()
    }

    // ─── Setters ────────────────────────────────────────────────────────────

    /// Replace the affine (nibabel style, 4×4).
    pub fn set_affine(&mut self, affine: Array2<f64>) {
        assert_eq!(affine.shape(), &[4, 4]);

        // Write sform
        for i in 0..4 {
            self.header.srow_x[i] = affine[[0, i]];
            self.header.srow_y[i] = affine[[1, i]];
            self.header.srow_z[i] = affine[[2, i]];
        }
        self.header.sform_code = 1; // scanner-based

        // Update pixdim from column norms
        let a = affine.slice(s![..3, ..3]);
        self.header.pixdim[1] = (a[[0, 0]].powi(2) + a[[1, 0]].powi(2) + a[[2, 0]].powi(2)).sqrt();
        self.header.pixdim[2] = (a[[0, 1]].powi(2) + a[[1, 1]].powi(2) + a[[2, 1]].powi(2)).sqrt();
        self.header.pixdim[3] = (a[[0, 2]].powi(2) + a[[1, 2]].powi(2) + a[[2, 2]].powi(2)).sqrt();
    }

    /// Set spacing (ITK style: [x, y, z]).
    pub fn set_spacing(&mut self, spacing: [f64; 3]) {
        assert!(spacing[0] > 0.0 && spacing[1] > 0.0 && spacing[2] > 0.0,
                "Spacing must be > 0");

        let mut aff = self.get_affine();
        let old = self.get_spacing();
        let r = aff.slice(s![..3, ..3]);

        // De-scale by old spacing, re-scale by new spacing (element-wise by column)
        let mut result = Array2::zeros((3, 3));
        for col in 0..3 {
            for row in 0..3 {
                result[[row, col]] = r[[row, col]] / old[col] * spacing[col];
            }
        }
        aff.slice_mut(s![..3, ..3]).assign(&result);
        self.set_affine(aff);
    }

    /// Set origin in LPS (ITK style: [x, y, z]).
    pub fn set_origin(&mut self, origin: [f64; 3]) {
        let mut aff = self.get_affine();
        // LPS → RAS
        aff[[0, 3]] = -origin[0];
        aff[[1, 3]] = -origin[1];
        aff[[2, 3]] = origin[2];
        self.set_affine(aff);
    }

    /// Set direction cosines in LPS (ITK style, 3×3).
    pub fn set_direction(&mut self, direction: [[f64; 3]; 3]) {
        // LPS → RAS (negate first two rows)
        let d_ras = [
            -direction[0][0], -direction[0][1], -direction[0][2],
            -direction[1][0], -direction[1][1], -direction[1][2],
             direction[2][0],  direction[2][1],  direction[2][2],
        ];

        let spacing = self.get_spacing();
        let mut aff = self.get_affine();

        // direction * diag(spacing)
        for col in 0..3 {
            for row in 0..3 {
                aff[[row, col]] = d_ras[row * 3 + col] * spacing[col];
            }
        }
        self.set_affine(aff);
    }

    /// Copy affine from another image (equivalent to copy-information).
    pub fn copy_information(&mut self, other: &Nifti1Image<T>) {
        self.set_affine(other.get_affine());
    }

    /// Reset to default header (identity transform, unit spacing).
    pub fn set_default_header(&mut self) {
        let default_aff = Array2::from_shape_vec((4, 4), vec![
            -1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]).unwrap();
        self.set_affine(default_aff);
    }

    // ─── Coordinate transforms ──────────────────────────────────────────────

    /// Voxel indices → physical coordinates (ITK style: [x, y, z] LPS).
    ///
    /// Uses the full affine matrix (direction + spacing + origin).
    pub fn ijk2xyz(&self, ijk: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let aff = self.get_affine();
        ijk.par_iter()
            .map(|&[i, j, k]| {
                // ITK indices [i,j,k] = [z,y,x]; nifti indices [x,y,z] = [k,j,i]
                let ras_x = aff[[0, 0]] * k + aff[[0, 1]] * j + aff[[0, 2]] * i + aff[[0, 3]];
                let ras_y = aff[[1, 0]] * k + aff[[1, 1]] * j + aff[[1, 2]] * i + aff[[1, 3]];
                let ras_z = aff[[2, 0]] * k + aff[[2, 1]] * j + aff[[2, 2]] * i + aff[[2, 3]];
                // RAS → LPS
                [-ras_x, -ras_y, ras_z]
            })
            .collect()
    }

    /// Physical coordinates → voxel indices (ITK style: [z, y, x]).
    pub fn xyz2ijk(&self, xyz: &[[f64; 3]]) -> Vec<[i32; 3]> {
        let aff = self.get_affine();
        let inv = affine_inv(aff.view());

        xyz.par_iter()
            .map(|&[lps_x, lps_y, lps_z]| {
                // LPS → RAS
                let ras = [-lps_x, -lps_y, lps_z, 1.0];
                let nifti_ijk = inv.dot(&Array1::from_vec(ras.to_vec()));
                // nifti [x,y,z] → ITK [z,y,x]
                let i = nifti_ijk[2].round() as i32;
                let j = nifti_ijk[1].round() as i32;
                let k = nifti_ijk[0].round() as i32;
                [i, j, k]
            })
            .collect()
    }
}

// ─── Debug ──────────────────────────────────────────────────────────────────

impl<T: NiftiType> fmt::Debug for Nifti1Image<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Size:     {:?}", self.get_size())?;
        writeln!(f, "Spacing:  {:?}", self.get_spacing())?;
        writeln!(f, "Origin:   {:?}", self.get_origin())?;
        writeln!(f, "Direction: {:?}", self.get_direction())
    }
}

// ─── Free functions ─────────────────────────────────────────────────────────

/// Read a NIfTI-1 image from disk.
pub fn read_image<T: NiftiType>(path: impl AsRef<Path>) -> Result<Nifti1Image<T>, NiftiError> {
    Nifti1Image::read(path)
}

/// Write a NIfTI-1 image to disk.
pub fn write_image<T: NiftiType>(
    im: &Nifti1Image<T>,
    path: impl AsRef<Path>,
) -> Result<(), NiftiError> {
    im.write(path)
}

/// Create a new image from an array and affine (nibabel style).
pub fn new<T: NiftiType>(
    ndarray: Array3<T>,
    affine: Array2<f64>,
) -> Result<Nifti1Image<T>, NiftiError> {
    // Build a minimal header from scratch
    let shape = ndarray.shape(); // [z, y, x] ITK
    let nx = shape[2] as i32;
    let ny = shape[1] as i32;
    let nz = shape[0] as i32;

    let header = Nifti1Header {
        little_endian: true,
        dim: [3, nx, ny, nz, 1, 0, 0, 0],
        datatype: T::DATATYPE,
        bitpix: T::BITPIX,
        pixdim: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        vox_offset: 352.0,
        srow_x: [0.0; 4],
        srow_y: [0.0; 4],
        srow_z: [0.0; 4],
        qform_code: 0,
        sform_code: 0,
        quatern_b: 0.0,
        quatern_c: 0.0,
        quatern_d: 0.0,
        qoffset_x: 0.0,
        qoffset_y: 0.0,
        qoffset_z: 0.0,
        magic: [b'n', b'+', b'1', b'\0'],
    };

    // Temporarily store as a mutable image to use set_affine
    let mut img = Nifti1Image { header, ndarray };
    img.set_affine(affine);
    Ok(img)
}

/// Create an image from an array with a default identity-like affine.
pub fn get_image_from_array<T: NiftiType>(
    ndarray: Array3<T>,
) -> Result<Nifti1Image<T>, NiftiError> {
    let affine = Array2::from_shape_vec((4, 4), vec![
        -1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]).unwrap();
    new(ndarray, affine)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal NIfTI-1 .nii file in memory (f32, 4×3×2 voxels).
    fn make_test_bytes() -> Vec<u8> {
        let nx = 4usize;
        let ny = 3usize;
        let nz = 2usize;

        // Data: 4*3*2 = 24 floats
        let data: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);

        let vox_offset = 352u32;
        let total = vox_offset as usize + data_bytes.len();
        let mut buf = vec![0u8; total];

        // sizeof_hdr
        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        // dim at OFF_DIM
        buf[OFF_DIM..OFF_DIM + 2].copy_from_slice(&(3i16).to_le_bytes());   // ndim
        buf[OFF_DIM + 2..OFF_DIM + 4].copy_from_slice(&(nx as i16).to_le_bytes());
        buf[OFF_DIM + 4..OFF_DIM + 6].copy_from_slice(&(ny as i16).to_le_bytes());
        buf[OFF_DIM + 6..OFF_DIM + 8].copy_from_slice(&(nz as i16).to_le_bytes());

        // datatype=16(f32), bitpix=32
        buf[OFF_DATATYPE..OFF_DATATYPE + 2].copy_from_slice(&16i16.to_le_bytes());
        buf[OFF_BITPIX..OFF_BITPIX + 2].copy_from_slice(&32i16.to_le_bytes());

        // pixdim: [1, 2.0, 3.0, 4.0, ...]
        buf[OFF_PIXDIM..OFF_PIXDIM + 4].copy_from_slice(&1.0f32.to_le_bytes());    // qfac
        buf[OFF_PIXDIM + 4..OFF_PIXDIM + 8].copy_from_slice(&2.0f32.to_le_bytes()); // dx
        buf[OFF_PIXDIM + 8..OFF_PIXDIM + 12].copy_from_slice(&3.0f32.to_le_bytes());// dy
        buf[OFF_PIXDIM + 12..OFF_PIXDIM + 16].copy_from_slice(&4.0f32.to_le_bytes());// dz

        // vox_offset
        buf[OFF_VOX_OFFSET..OFF_VOX_OFFSET + 4].copy_from_slice(&(vox_offset as f32).to_le_bytes());

        // sform: identity-like (RAS)
        buf[OFF_SROW_X..OFF_SROW_X + 4].copy_from_slice(&2.0f32.to_le_bytes());
        buf[OFF_SROW_X + 4..OFF_SROW_X + 8].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_X + 8..OFF_SROW_X + 12].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_X + 12..OFF_SROW_X + 16].copy_from_slice(&0.0f32.to_le_bytes());

        buf[OFF_SROW_Y..OFF_SROW_Y + 4].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_Y + 4..OFF_SROW_Y + 8].copy_from_slice(&3.0f32.to_le_bytes());
        buf[OFF_SROW_Y + 8..OFF_SROW_Y + 12].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_Y + 12..OFF_SROW_Y + 16].copy_from_slice(&0.0f32.to_le_bytes());

        buf[OFF_SROW_Z..OFF_SROW_Z + 4].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_Z + 4..OFF_SROW_Z + 8].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_SROW_Z + 8..OFF_SROW_Z + 12].copy_from_slice(&4.0f32.to_le_bytes());
        buf[OFF_SROW_Z + 12..OFF_SROW_Z + 16].copy_from_slice(&0.0f32.to_le_bytes());

        // sform_code = 1
        buf[OFF_SFORM_CODE..OFF_SFORM_CODE + 2].copy_from_slice(&1i16.to_le_bytes());

        // magic
        buf[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(b"n+1\0");

        // Also set qform_code = 1 so it's non-zero
        buf[OFF_QFORM_CODE..OFF_QFORM_CODE + 2].copy_from_slice(&1i16.to_le_bytes());
        // Set qoffset values for the origin
        buf[OFF_QOFFSET_X..OFF_QOFFSET_X + 4].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_QOFFSET_Y..OFF_QOFFSET_Y + 4].copy_from_slice(&0.0f32.to_le_bytes());
        buf[OFF_QOFFSET_Z..OFF_QOFFSET_Z + 4].copy_from_slice(&0.0f32.to_le_bytes());

        // data at offset 352
        buf[vox_offset as usize..][..data_bytes.len()].copy_from_slice(data_bytes);

        buf
    }

    #[test]
    fn test_read_nifti_from_bytes() {
        let bytes = make_test_bytes();
        let img = Nifti1Image::<f32>::from_bytes(&bytes).unwrap();

        assert_eq!(img.get_size(), [4, 3, 2]);
        let spacing = img.get_spacing();
        assert!((spacing[0] - 2.0).abs() < 1e-6);
        assert!((spacing[1] - 3.0).abs() < 1e-6);
        assert!((spacing[2] - 4.0).abs() < 1e-6);

        // Check data values (ITK [z,y,x] order)
        let arr = img.ndarray();
        eprintln!("arr = {:?}", arr);
        eprintln!("arr.strides() = {:?}", arr.strides());
        assert_eq!(arr.shape(), [2, 3, 4]);
        // voxel [z=0, y=0, x=0] = data[0] in file order
        assert!((arr[[0, 0, 0]] - 0.0).abs() < 1e-6);
        // voxel [z=0, y=0, x=1] = data[1] in file order
        assert!((arr[[0, 0, 1]] - 1.0).abs() < 1e-6);
        // voxel [z=1, y=0, x=0] = data[12] in file order (nx*ny = 12)
        assert!((arr[[1, 0, 0]] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_round_trip() {
        let bytes = make_test_bytes();
        let img = Nifti1Image::<f32>::from_bytes(&bytes).unwrap();

        // Write back
        let written = img.to_bytes().unwrap();

        // Re-read
        let img2 = Nifti1Image::<f32>::from_bytes(&written).unwrap();

        assert_eq!(img2.get_size(), [4, 3, 2]);
        let s2 = img2.get_spacing();
        assert!((s2[0] - 2.0).abs() < 1e-5);
        assert!((s2[1] - 3.0).abs() < 1e-5);
        assert!((s2[2] - 4.0).abs() < 1e-5);

        // Data should match exactly
        assert_eq!(img.ndarray(), img2.ndarray());
    }

    #[test]
    fn test_affine_round_trip() {
        let bytes = make_test_bytes();
        let mut img = Nifti1Image::<f32>::from_bytes(&bytes).unwrap();

        let aff_orig = img.get_affine();
        assert!((aff_orig[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((aff_orig[[1, 1]] - 3.0).abs() < 1e-6);
        assert!((aff_orig[[2, 2]] - 4.0).abs() < 1e-6);

        // Set new affine
        let new_aff = Array2::from_shape_vec((4, 4), vec![
            1.0, 0.0, 0.0, 10.0,
            0.0, 2.0, 0.0, 20.0,
            0.0, 0.0, 3.0, 30.0,
            0.0, 0.0, 0.0, 1.0,
        ]).unwrap();
        img.set_affine(new_aff.clone());

        let read_back = img.get_affine();
        for i in 0..4 {
            for j in 0..4 {
                assert!((read_back[[i, j]] - new_aff[[i, j]]).abs() < 1e-10,
                    "mismatch at ({i},{j}): {} vs {}", read_back[[i,j]], new_aff[[i,j]]);
            }
        }
    }

    #[test]
    fn test_origin_direction() {
        let bytes = make_test_bytes();
        let img = Nifti1Image::<f32>::from_bytes(&bytes).unwrap();

        // sform is identity-like → origin is [0,0,0]
        let origin = img.get_origin();
        assert!((origin[0]).abs() < 1e-6);
        assert!((origin[1]).abs() < 1e-6);
        assert!((origin[2]).abs() < 1e-6);

        // direction in LPS: first two rows negated vs RAS identity
        let dir = img.get_direction();
        // RAS identity [[1,0,0],[0,1,0],[0,0,1]] → LPS [[-1,0,0],[0,-1,0],[0,0,1]]
        assert!((dir[0][0] + 1.0).abs() < 1e-6);
        assert!((dir[1][1] + 1.0).abs() < 1e-6);
        assert!((dir[2][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ijk2xyz_identity() {
        let bytes = make_test_bytes();
        let img = Nifti1Image::<f32>::from_bytes(&bytes).unwrap();

        // With identity direction, ijk2xyz should give [o_x + k*dx, o_y + j*dy, o_z + i*dz]
        // where [i,j,k] = [z,y,x]
        let result = img.ijk2xyz(&[[0.0, 0.0, 0.0]]);
        assert!((result[0][0]).abs() < 1e-6);
        assert!((result[0][1]).abs() < 1e-6);
        assert!((result[0][2]).abs() < 1e-6);

        // voxel [i=0,j=0,k=1] = ITK [z=0,y=0,x=1]
        // RAS = [1*2, 0, 0] = [2, 0, 0], LPS = [-2, 0, 0]
        let result = img.ijk2xyz(&[[0.0, 0.0, 1.0]]);
        assert!((result[0][0] + 2.0).abs() < 1e-6);
        assert!((result[0][1]).abs() < 1e-6);
        assert!((result[0][2]).abs() < 1e-6);

        // Round-trip
        let ijk = img.xyz2ijk(&result);
        assert_eq!(ijk[0], [0, 0, 1]);
    }
}
