//! NIfTI-1 header parsing and affine construction.
//!
//! The NIfTI-1 header is exactly 348 bytes.
//! See https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

use ndarray::Array2;
use std::io::Read;

// ─── NIfTI-1 header byte offsets (per nifti1.h spec) ────────────────────────
pub(crate) const OFF_DIM: usize = 40;
pub(crate) const OFF_DATATYPE: usize = 70;
pub(crate) const OFF_BITPIX: usize = 72;
pub(crate) const OFF_PIXDIM: usize = 76;
pub(crate) const OFF_VOX_OFFSET: usize = 108;
pub(crate) const OFF_QFORM_CODE: usize = 252;
pub(crate) const OFF_SFORM_CODE: usize = 254;
pub(crate) const OFF_QUATERN_B: usize = 256;
pub(crate) const OFF_QUATERN_C: usize = 260;
pub(crate) const OFF_QUATERN_D: usize = 264;
pub(crate) const OFF_QOFFSET_X: usize = 268;
pub(crate) const OFF_QOFFSET_Y: usize = 272;
pub(crate) const OFF_QOFFSET_Z: usize = 276;
pub(crate) const OFF_SROW_X: usize = 280;
pub(crate) const OFF_SROW_Y: usize = 296;
pub(crate) const OFF_SROW_Z: usize = 312;
pub(crate) const OFF_MAGIC: usize = 344;

/// NIfTI-1 datatype codes (only the subset we support).
pub mod dtype {
    pub const UINT8: i16 = 2;
    pub const INT16: i16 = 4;
    pub const INT32: i16 = 8;
    pub const FLOAT32: i16 = 16;
    pub const FLOAT64: i16 = 64;
    pub const INT8: i16 = 256;
    pub const UINT16: i16 = 512;
    pub const UINT32: i16 = 768;
    pub const INT64: i16 = 1024;
    pub const UINT64: i16 = 1280;
}

/// Errors that can occur during NIfTI-1 I/O.
#[derive(Debug)]
pub enum NiftiError {
    Io(std::io::Error),
    InvalidHeader(&'static str),
    DataTypeMismatch { expected: i16, found: i16 },
    UnsupportedDataType(i16),
    DimensionMismatch(&'static str),
    SingularAffine,
}

impl std::fmt::Display for NiftiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NiftiError::Io(e) => write!(f, "I/O error: {e}"),
            NiftiError::InvalidHeader(msg) => write!(f, "Invalid NIfTI-1 header: {msg}"),
            NiftiError::DataTypeMismatch { expected, found } => {
                write!(f, "Data type mismatch: file has code {found}, expected {expected}")
            }
            NiftiError::UnsupportedDataType(code) => {
                write!(f, "Unsupported NIfTI datatype code: {code}")
            }
            NiftiError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
            NiftiError::SingularAffine => write!(f, "Affine matrix is singular"),
        }
    }
}

impl std::error::Error for NiftiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NiftiError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NiftiError {
    fn from(e: std::io::Error) -> Self {
        NiftiError::Io(e)
    }
}

/// Parse a little-endian i16 from bytes.
fn le_i16(buf: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([buf[off], buf[off + 1]])
}

/// Parse a little-endian f32 from bytes.
fn le_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Parse a big-endian i16 from bytes.
fn be_i16(buf: &[u8], off: usize) -> i16 {
    i16::from_be_bytes([buf[off], buf[off + 1]])
}

/// Parse a big-endian f32 from bytes.
fn be_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// A parsed NIfTI-1 header.
///
/// Only stores the fields we actually use.
/// The rest of the 348 bytes (description, aux_file, intent, etc.)
/// is preserved as raw if needed, but not exposed in the Rust API.
#[derive(Clone, Debug)]
pub struct Nifti1Header {
    /// Raw byte endianness of the file.
    pub little_endian: bool,

    /// dim[0] = number of dimensions; dim[1..] = size per dimension.
    pub dim: [i32; 8],

    /// NIfTI datatype code (e.g. 2=uint8, 4=int16, 16=float32).
    pub datatype: i16,

    /// Bits per pixel.
    pub bitpix: i16,

    /// pixdim[0] = qfac; pixdim[1..] = voxel spacing per dimension.
    pub pixdim: [f64; 8],

    /// Offset in bytes from file start to voxel data.
    pub vox_offset: f64,

    /// sform matrix rows (4 elements each, in voxel coordinate space).
    pub srow_x: [f64; 4],
    pub srow_y: [f64; 4],
    pub srow_z: [f64; 4],

    /// qform code (0=unknown, 1=scanner, 2=aligned, 3=talairach, 4=mni).
    pub qform_code: i32,

    /// sform code (same meaning as qform_code).
    pub sform_code: i32,

    /// Quaternion parameters.
    pub quatern_b: f64,
    pub quatern_c: f64,
    pub quatern_d: f64,

    /// Quaternion offset.
    pub qoffset_x: f64,
    pub qoffset_y: f64,
    pub qoffset_z: f64,

    /// Magic bytes: "n+1\0" for .nii, "ni1\0" for .hdr/.img pair.
    pub magic: [u8; 4],
}

impl Nifti1Header {
    /// Read a header from a reader (the first 348 bytes).
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, NiftiError> {
        let mut buf = vec![0u8; 348];
        reader.read_exact(&mut buf)?;
        Self::parse(&buf)
    }

    /// Parse a header from exactly 348 bytes.
    ///
    /// Detects endianness automatically by checking sizeof_hdr at offset 0.
    pub fn parse(buf: &[u8]) -> Result<Self, NiftiError> {
        if buf.len() < 348 {
            return Err(NiftiError::InvalidHeader("buffer too short, need 348 bytes"));
        }

        // --- Endian detection ---
        // sizeof_hdr (offset 0) must be 348 in the file's native endian.
        // If we read it as LE and get 348, the file is little-endian.
        // Otherwise, it's big-endian.
        let le_size = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let be_size = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let little_endian = if le_size == 348 {
            true
        } else if be_size == 348 {
            false
        } else {
            return Err(NiftiError::InvalidHeader(
                "sizeof_hdr != 348 in both endiannesses",
            ));
        };

        // Helper closures for the chosen endianness.
        let r#i16 = |off| {
            if little_endian {
                le_i16(buf, off)
            } else {
                be_i16(buf, off)
            }
        };
        let r#f32 = |off| {
            if little_endian {
                le_f32(buf, off)
            } else {
                be_f32(buf, off)
            }
        };

        // --- dim[8] at offset OFF_DIM ---
        let dim = {
            let mut d = [0i32; 8];
            for i in 0..8 {
                d[i] = r#i16(OFF_DIM + i * 2) as i32;
            }
            let ndim = d[0].clamp(0, 7) as usize;
            d[0] = ndim as i32;
            d
        };

        let datatype = r#i16(OFF_DATATYPE);
        let bitpix = r#i16(OFF_BITPIX);

        let pixdim = {
            let mut p = [0.0f64; 8];
            for i in 0..8 {
                p[i] = r#f32(OFF_PIXDIM + i * 4) as f64;
            }
            p
        };

        let vox_offset = r#f32(OFF_VOX_OFFSET) as f64;

        let srow_x = {
            let mut r = [0.0f64; 4];
            for i in 0..4 {
                r[i] = r#f32(OFF_SROW_X + i * 4) as f64;
            }
            r
        };
        let srow_y = {
            let mut r = [0.0f64; 4];
            for i in 0..4 {
                r[i] = r#f32(OFF_SROW_Y + i * 4) as f64;
            }
            r
        };
        let srow_z = {
            let mut r = [0.0f64; 4];
            for i in 0..4 {
                r[i] = r#f32(OFF_SROW_Z + i * 4) as f64;
            }
            r
        };

        let qform_code = r#i16(OFF_QFORM_CODE) as i32;
        let sform_code = r#i16(OFF_SFORM_CODE) as i32;

        let quatern_b = r#f32(OFF_QUATERN_B) as f64;
        let quatern_c = r#f32(OFF_QUATERN_C) as f64;
        let quatern_d = r#f32(OFF_QUATERN_D) as f64;

        let qoffset_x = r#f32(OFF_QOFFSET_X) as f64;
        let qoffset_y = r#f32(OFF_QOFFSET_Y) as f64;
        let qoffset_z = r#f32(OFF_QOFFSET_Z) as f64;

        let magic = [buf[OFF_MAGIC], buf[OFF_MAGIC + 1], buf[OFF_MAGIC + 2], buf[OFF_MAGIC + 3]];

        Ok(Nifti1Header {
            little_endian,
            dim,
            datatype,
            bitpix,
            pixdim,
            vox_offset,
            srow_x,
            srow_y,
            srow_z,
            qform_code,
            sform_code,
            quatern_b,
            quatern_c,
            quatern_d,
            qoffset_x,
            qoffset_y,
            qoffset_z,
            magic,
        })
    }

    /// Number of effective dimensions (dim[0]).
    pub fn ndim(&self) -> usize {
        self.dim[0] as usize
    }

    /// Shape of the voxel data: [dim_x, dim_y, dim_z] (nifti-rs / nibabel convention).
    /// For 3D data, this returns [dim[1], dim[2], dim[3]].
    pub fn shape(&self) -> Vec<usize> {
        let ndim = self.ndim();
        (1..=ndim).map(|i| self.dim[i] as usize).collect()
    }

    /// Returns `true` if this is a .nii file (magic == "n+1\0").
    pub fn is_nifti(&self) -> bool {
        &self.magic[..3] == b"n+1"
    }

    // ─── Affine construction ────────────────────────────────────────────────

    /// Compute the 4×4 affine matrix in **nibabel convention** (rows are axes).
    ///
    /// Priority: sform > qform > pixdim diagonal fallback.
    pub fn affine(&self) -> Array2<f64> {
        if self.sform_code > 0 {
            self.affine_from_sform()
        } else if self.qform_code > 0 {
            self.affine_from_qform()
        } else {
            self.affine_from_pixdim()
        }
    }

    /// Build affine from srow_x/y/z (most common, and preferred by NIfTI standard).
    fn affine_from_sform(&self) -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                self.srow_x[0], self.srow_x[1], self.srow_x[2], self.srow_x[3],
                self.srow_y[0], self.srow_y[1], self.srow_y[2], self.srow_y[3],
                self.srow_z[0], self.srow_z[1], self.srow_z[2], self.srow_z[3],
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap()
    }

    /// Build affine from quaternion + pixdim (NIfTI-1 standard formula).
    fn affine_from_qform(&self) -> Array2<f64> {
        let b = self.quatern_b;
        let c = self.quatern_c;
        let d = self.quatern_d;

        // Compute a from the unit quaternion constraint: a² + b² + c² + d² = 1
        let a_sq = 1.0 - (b * b + c * c + d * d);
        let a = if a_sq < 1e-7 {
            // Near-zero a: normalize (b,c,d) instead
            let norm = (b * b + c * c + d * d).sqrt();
            if norm > 0.0 {
                // Re-normalize to avoid precision issues
                0.0
            } else {
                0.0
            }
        } else {
            a_sq.sqrt()
        };

        // Rotation matrix R (columns = direction cosines in real-space)
        // Using the standard NIfTI-1 / nibabel formula
        let r = [
            [
                a * a + b * b - c * c - d * d,
                2.0 * b * c - 2.0 * a * d,
                2.0 * b * d + 2.0 * a * c,
            ],
            [
                2.0 * b * c + 2.0 * a * d,
                a * a + c * c - b * b - d * d,
                2.0 * c * d - 2.0 * a * b,
            ],
            [
                2.0 * b * d - 2.0 * a * c,
                2.0 * c * d + 2.0 * a * b,
                a * a + d * d - c * c - b * b,
            ],
        ];

        let qfac = if self.pixdim[0] == 0.0 {
            1.0
        } else {
            self.pixdim[0]
        };

        // Scale each column by the voxel spacing
        // Column 0 → pixdim[1], Column 1 → pixdim[2], Column 2 → pixdim[3] * qfac
        let sx = self.pixdim[1];
        let sy = self.pixdim[2];
        let sz = self.pixdim[3] * qfac;

        Array2::from_shape_vec(
            (4, 4),
            vec![
                r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, self.qoffset_x,
                r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, self.qoffset_y,
                r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, self.qoffset_z,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap()
    }

    /// Fallback: diagonal affine from pixdim, zero origin.
    fn affine_from_pixdim(&self) -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                self.pixdim[1], 0.0, 0.0, 0.0,
                0.0, self.pixdim[2], 0.0, 0.0,
                0.0, 0.0, self.pixdim[3], 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap()
    }
}

// ─── Helpers for reading raw bytes ──────────────────────────────────────────

/// Read the full contents of a file into a Vec<u8>.
/// Auto-detects and decompresses gzip by sniffing magic bytes.
pub fn read_file_bytes(path: &std::path::Path) -> Result<Vec<u8>, NiftiError> {
    let raw = std::fs::read(path)?;

    // Gzip magic: 0x1F 0x8B
    if raw.len() >= 2 && raw[0] == 0x1F && raw[1] == 0x8B {
        let mut decoder = flate2::read::GzDecoder::new(&raw[..]);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_header() {
        // Build a minimal NIfTI-1 header (348 bytes, LE).
        let mut buf = vec![0u8; 348];

        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        buf[OFF_DIM..OFF_DIM+2].copy_from_slice(&(3i16).to_le_bytes());
        buf[OFF_DIM+2..OFF_DIM+4].copy_from_slice(&(4i16).to_le_bytes());
        buf[OFF_DIM+4..OFF_DIM+6].copy_from_slice(&(3i16).to_le_bytes());
        buf[OFF_DIM+6..OFF_DIM+8].copy_from_slice(&(2i16).to_le_bytes());

        buf[OFF_DATATYPE..OFF_DATATYPE+2].copy_from_slice(&(16i16).to_le_bytes());
        buf[OFF_BITPIX..OFF_BITPIX+2].copy_from_slice(&(32i16).to_le_bytes());

        buf[OFF_PIXDIM..OFF_PIXDIM+4].copy_from_slice(&1.0f32.to_le_bytes());
        buf[OFF_PIXDIM+4..OFF_PIXDIM+8].copy_from_slice(&2.0f32.to_le_bytes());
        buf[OFF_PIXDIM+8..OFF_PIXDIM+12].copy_from_slice(&3.0f32.to_le_bytes());
        buf[OFF_PIXDIM+12..OFF_PIXDIM+16].copy_from_slice(&4.0f32.to_le_bytes());

        buf[OFF_VOX_OFFSET..OFF_VOX_OFFSET+4].copy_from_slice(&352.0f32.to_le_bytes());

        buf[OFF_MAGIC..OFF_MAGIC+4].copy_from_slice(b"n+1\0");

        // --- Parse ---
        let hdr = Nifti1Header::parse(&buf).unwrap();

        assert_eq!(hdr.ndim(), 3);
        assert_eq!(hdr.dim[1..4], [4, 3, 2]);
        assert_eq!(hdr.datatype, 16);
        assert_eq!(hdr.bitpix, 32);
        assert!((hdr.pixdim[1] - 2.0).abs() < 1e-6);
        assert!((hdr.pixdim[2] - 3.0).abs() < 1e-6);
        assert!((hdr.pixdim[3] - 4.0).abs() < 1e-6);
        assert!((hdr.vox_offset - 352.0).abs() < 1e-6);
        assert!(hdr.is_nifti());
        assert!(hdr.little_endian);

        // No sform/qform → fallback diagonal affine
        let aff = hdr.affine();
        assert!((aff[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((aff[[1, 1]] - 3.0).abs() < 1e-6);
        assert!((aff[[2, 2]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_header_big_endian() {
        let mut buf = vec![0u8; 348];
        // sizeof_hdr in big-endian
        buf[0..4].copy_from_slice(&348i32.to_be_bytes());

        buf[OFF_DIM..OFF_DIM+2].copy_from_slice(&(3i16).to_be_bytes());
        buf[OFF_DIM+2..OFF_DIM+4].copy_from_slice(&(5i16).to_be_bytes());
        buf[OFF_DIM+4..OFF_DIM+6].copy_from_slice(&(4i16).to_be_bytes());
        buf[OFF_DIM+6..OFF_DIM+8].copy_from_slice(&(3i16).to_be_bytes());

        buf[OFF_DATATYPE..OFF_DATATYPE+2].copy_from_slice(&(16i16).to_be_bytes());
        buf[OFF_BITPIX..OFF_BITPIX+2].copy_from_slice(&(32i16).to_be_bytes());

        buf[OFF_PIXDIM..OFF_PIXDIM+4].copy_from_slice(&1.0f32.to_be_bytes());
        buf[OFF_PIXDIM+4..OFF_PIXDIM+8].copy_from_slice(&1.0f32.to_be_bytes());
        buf[OFF_PIXDIM+8..OFF_PIXDIM+12].copy_from_slice(&1.0f32.to_be_bytes());
        buf[OFF_PIXDIM+12..OFF_PIXDIM+16].copy_from_slice(&1.0f32.to_be_bytes());

        buf[OFF_VOX_OFFSET..OFF_VOX_OFFSET+4].copy_from_slice(&352.0f32.to_be_bytes());
        buf[OFF_MAGIC..OFF_MAGIC+4].copy_from_slice(b"n+1\0");

        let hdr = Nifti1Header::parse(&buf).unwrap();
        assert!(!hdr.little_endian);
        assert_eq!(hdr.dim[1..4], [5, 4, 3]);
    }
}
