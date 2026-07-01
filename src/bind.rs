//! Python bindings for nii-rs.
//!
//! A single `Nifti1Image` pyclass handles all voxel data types internally,
//! replacing the previous macro-generated 10× class explosion.

use crate::header::{Nifti1Header, NiftiError, read_file_bytes, dtype};
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::path::Path;

impl From<NiftiError> for pyo3::PyErr {
    fn from(e: NiftiError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

// ─── Type-erased voxel data ────────────────────────────────────────────────

#[derive(Clone)]
enum ImageData {
    F32(Array3<f32>),
    F64(Array3<f64>),
    U8(Array3<u8>),
    U16(Array3<u16>),
    U32(Array3<u32>),
    U64(Array3<u64>),
    I8(Array3<i8>),
    I16(Array3<i16>),
    I32(Array3<i32>),
    I64(Array3<i64>),
}

fn elem_size(code: i16) -> usize {
    match code {
        dtype::UINT8 | dtype::INT8 => 1,
        dtype::UINT16 | dtype::INT16 => 2,
        dtype::FLOAT32 | dtype::UINT32 | dtype::INT32 => 4,
        dtype::FLOAT64 | dtype::UINT64 | dtype::INT64 => 8,
        _ => 4,
    }
}

fn read_data(bytes: &[u8], hdr: &Nifti1Header) -> Result<ImageData, NiftiError> {
    let shape = hdr.shape();
    if shape.len() < 3 {
        return Err(NiftiError::DimensionMismatch("expected >=3 dims"));
    }
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let off = hdr.vox_offset as usize;
    let es = elem_size(hdr.datatype);
    let n = nx * ny * nz;
    let raw = &bytes[off..off + n * es];

    // The [x,y,z] → [z,y,x] reorder is a sequential copy when iterating
    // in [z,y,x] order, because nx*ny == ny*nx.  So we skip the intermediate
    // byte-level buffer and write typed data directly.
    macro_rules! read_typed {
        ($t:ty) => {{
            let flat: &[$t] = bytemuck::cast_slice(raw);
            Array3::from_shape_vec((nz, ny, nx), flat.to_vec())
                .map_err(|_| NiftiError::DimensionMismatch("shape"))?
        }};
    }
    Ok(match hdr.datatype {
        dtype::FLOAT32 => ImageData::F32(read_typed!(f32)),
        dtype::FLOAT64 => ImageData::F64(read_typed!(f64)),
        dtype::UINT8   => ImageData::U8( read_typed!(u8) ),
        dtype::INT8    => ImageData::I8( read_typed!(i8) ),
        dtype::UINT16  => ImageData::U16(read_typed!(u16)),
        dtype::INT16   => ImageData::I16(read_typed!(i16)),
        dtype::UINT32  => ImageData::U32(read_typed!(u32)),
        dtype::INT32   => ImageData::I32(read_typed!(i32)),
        dtype::INT64   => ImageData::I64(read_typed!(i64)),
        dtype::UINT64  => ImageData::U64(read_typed!(u64)),
        code => return Err(NiftiError::UnsupportedDataType(code)),
    })
}

fn write_data(data: &ImageData) -> Vec<u8> {
    match data {
        ImageData::F32(a) => to_file_order(a),
        ImageData::F64(a) => to_file_order(a),
        ImageData::U8(a)  => to_file_order(a),
        ImageData::U16(a) => to_file_order(a),
        ImageData::U32(a) => to_file_order(a),
        ImageData::U64(a) => to_file_order(a),
        ImageData::I8(a)  => to_file_order(a),
        ImageData::I16(a) => to_file_order(a),
        ImageData::I32(a) => to_file_order(a),
        ImageData::I64(a) => to_file_order(a),
    }
}

fn read_typed<T: bytemuck::Pod>(bytes: &[u8], nz: usize, ny: usize, nx: usize) -> Result<Array3<T>, pyo3::PyErr> {
    let flat: &[T] = bytemuck::cast_slice(bytes);
    Array3::from_shape_vec((nz, ny, nx), flat.to_vec())
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape mismatch"))
}

/// Read a numpy array's data buffer directly, one copy to Vec.
fn read_numpy_array(
    arr: &Bound<'_, PyAny>,
    dtype_s: &str,
    nz: usize, ny: usize, nx: usize,
) -> PyResult<ImageData> {
    macro_rules! try_downcast {
        ($py3:ty, $var:ident) => {{
            if let Ok(py_arr) = arr.downcast::<numpy::PyArray3<$py3>>() {
                let data = py_arr.readonly().as_array().to_owned();
                return Ok(ImageData::$var(data));
            }
        }};
    }
    try_downcast!(f32, F32);
    try_downcast!(f64, F64);
    try_downcast!(u8,  U8);
    try_downcast!(i8,  I8);
    try_downcast!(u16, U16);
    try_downcast!(i16, I16);
    try_downcast!(u32, U32);
    try_downcast!(i32, I32);
    try_downcast!(u64, U64);
    try_downcast!(i64, I64);

    // Fallback: in case downcast fails (e.g. non-contiguous array),
    // use tobytes() which handles all cases.
    let bytes = arr.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    match dtype_s {
        "float32" => Ok(ImageData::F32(read_typed::<f32>(&bytes, nz, ny, nx)?)),
        "float64" => Ok(ImageData::F64(read_typed::<f64>(&bytes, nz, ny, nx)?)),
        "uint8"   => Ok(ImageData::U8(Array3::from_shape_vec((nz, ny, nx), bytes)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape"))?)),
        "int8"    => Ok(ImageData::I8(read_typed::<i8>(&bytes, nz, ny, nx)?)),
        "uint16"  => Ok(ImageData::U16(read_typed::<u16>(&bytes, nz, ny, nx)?)),
        "int16"   => Ok(ImageData::I16(read_typed::<i16>(&bytes, nz, ny, nx)?)),
        "uint32"  => Ok(ImageData::U32(read_typed::<u32>(&bytes, nz, ny, nx)?)),
        "int32"   => Ok(ImageData::I32(read_typed::<i32>(&bytes, nz, ny, nx)?)),
        "uint64"  => Ok(ImageData::U64(read_typed::<u64>(&bytes, nz, ny, nx)?)),
        "int64"   => Ok(ImageData::I64(read_typed::<i64>(&bytes, nz, ny, nx)?)),
        other     => Err(pyo3::exceptions::PyValueError::new_err(
            format!("unsupported dtype: {other}"))),
    }
}

/// Extract NIfTI datatype code and bitpix from ImageData.
fn dtype_bitpix(data: &ImageData) -> (i16, i16) {
    match data {
        ImageData::F32(_) => (dtype::FLOAT32, 32),
        ImageData::F64(_) => (dtype::FLOAT64, 64),
        ImageData::U8(_)  => (dtype::UINT8, 8),
        ImageData::U16(_) => (dtype::UINT16, 16),
        ImageData::U32(_) => (dtype::UINT32, 32),
        ImageData::U64(_) => (dtype::UINT64, 64),
        ImageData::I8(_)  => (dtype::INT8, 8),
        ImageData::I16(_) => (dtype::INT16, 16),
        ImageData::I32(_) => (dtype::INT32, 32),
        ImageData::I64(_) => (dtype::INT64, 64),
    }
}

fn to_file_order<T: bytemuck::Pod>(arr: &Array3<T>) -> Vec<u8> {
    let shape = arr.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    let es = std::mem::size_of::<T>();
    let n = nx * ny * nz;
    let mut out = vec![0u8; n * es];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = (x + nx * y + nx * ny * z) * es;
                let val: &[u8] = bytemuck::bytes_of(&arr[[z, y, x]]);
                out[dst..dst + es].copy_from_slice(val);
            }
        }
    }
    out
}

// ─── Python class ──────────────────────────────────────────────────────────

#[pyclass(name = "Nifti1Image")]
#[derive(Clone)]
pub struct PyNifti1Image {
    header: Nifti1Header,
    data: ImageData,
}

#[pymethods]
impl PyNifti1Image {
    #[staticmethod]
    fn read(path: &str) -> PyResult<Self> {
        let bytes = read_file_bytes(Path::new(path))?;
        let header = Nifti1Header::parse(&bytes)?;
        let data = read_data(&bytes, &header)?;
        Ok(PyNifti1Image { header, data })
    }

    fn write(&self, path: &str) -> PyResult<()> {
        let is_gz = path.ends_with(".gz");
        let bytes = self.to_bytes();
        if is_gz {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::Write;
            let mut enc = GzEncoder::new(Vec::new(), Compression::new(1));
            enc.write_all(&bytes)?;
            std::fs::write(path, enc.finish()?)?;
        } else {
            std::fs::write(path, bytes)?;
        }
        Ok(())
    }

    // ─── String representation ──────────────────────────────────────────────

    fn __str__(&self) -> String {
        let s = self.get_size();
        let sp = self.get_spacing();
        let o = self.get_origin();
        let d = self.get_direction();
        format!(
            "Size: {:?}\nSpacing: {:?}\nOrigin: {:?}\nDirection: {:?}",
            s, sp, o, d
        )
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    pub fn get_size(&self) -> [u32; 3] {
        let s = arr_shape(&self.data);
        [s[2] as u32, s[1] as u32, s[0] as u32]
    }

    pub fn get_spacing(&self) -> [f64; 3] {
        let h = &self.header;
        [h.pixdim[1], h.pixdim[2], h.pixdim[3]]
    }

    pub fn get_origin(&self) -> [f64; 3] {
        let aff = self.header.affine();
        [-aff[[0, 3]], -aff[[1, 3]], aff[[2, 3]]]
    }

    pub fn get_direction(&self) -> [[f64; 3]; 3] {
        let aff = self.header.affine();
        let a = aff.slice(s![..3, ..3]);
        let sx = (a[[0,0]].powi(2)+a[[1,0]].powi(2)+a[[2,0]].powi(2)).sqrt();
        let sy = (a[[0,1]].powi(2)+a[[1,1]].powi(2)+a[[2,1]].powi(2)).sqrt();
        let sz = (a[[0,2]].powi(2)+a[[1,2]].powi(2)+a[[2,2]].powi(2)).sqrt();
        let d = [
            [a[[0,0]]/sx, a[[0,1]]/sy, a[[0,2]]/sz],
            [a[[1,0]]/sx, a[[1,1]]/sy, a[[1,2]]/sz],
            [a[[2,0]]/sx, a[[2,1]]/sy, a[[2,2]]/sz],
        ];
        [[-d[0][0], -d[0][1], -d[0][2]],
         [-d[1][0], -d[1][1], -d[1][2]],
         [ d[2][0],  d[2][1],  d[2][2]]]
    }

    pub fn get_unit_size(&self) -> f64 {
        let s = self.get_spacing();
        s[0] * s[1] * s[2]
    }

    fn get_affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.header.affine().into_pyarray(py)
    }

    pub fn ndarray<'py>(&self, py: Python<'py>) -> PyObject {
        match &self.data {
            ImageData::F32(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::F64(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::U8(a)  => a.clone().into_pyarray(py).into_any().into(),
            ImageData::U16(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::U32(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::U64(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::I8(a)  => a.clone().into_pyarray(py).into_any().into(),
            ImageData::I16(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::I32(a) => a.clone().into_pyarray(py).into_any().into(),
            ImageData::I64(a) => a.clone().into_pyarray(py).into_any().into(),
        }
    }

    /// Return voxel data as a numpy ndarray, consuming the image.
    ///
    /// After calling this, the image is left with an empty array.
    /// Prefer ``ndarray()`` if you need the image afterward.
    pub fn into_ndarray<'py>(&mut self, py: Python<'py>) -> PyObject {
        let dummy = ImageData::F32(Array3::from_shape_vec((0, 0, 0), vec![]).unwrap());
        let data = std::mem::replace(&mut self.data, dummy);
        match data {
            ImageData::F32(a) => a.into_pyarray(py).into_any().into(),
            ImageData::F64(a) => a.into_pyarray(py).into_any().into(),
            ImageData::U8(a)  => a.into_pyarray(py).into_any().into(),
            ImageData::U16(a) => a.into_pyarray(py).into_any().into(),
            ImageData::U32(a) => a.into_pyarray(py).into_any().into(),
            ImageData::U64(a) => a.into_pyarray(py).into_any().into(),
            ImageData::I8(a)  => a.into_pyarray(py).into_any().into(),
            ImageData::I16(a) => a.into_pyarray(py).into_any().into(),
            ImageData::I32(a) => a.into_pyarray(py).into_any().into(),
            ImageData::I64(a) => a.into_pyarray(py).into_any().into(),
        }
    }

    // ─── Constructors ───────────────────────────────────────────────────────

    #[staticmethod]
    fn new<'py>(arr: &Bound<'py, PyAny>, affine: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let dtype_s = arr.getattr("dtype")?.str()?.to_string();
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("expected 3D array"));
        }
        let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
        let aff = affine.as_array().to_owned();

        let data = read_numpy_array(arr, &dtype_s, nz, ny, nx)?;

        let (dt, bp) = dtype_bitpix(&data);
        let mut hdr = Nifti1Header {
            little_endian: true,
            dim: [3, nx as i32, ny as i32, nz as i32, 1, 0, 0, 0],
            datatype: dt,
            bitpix: bp,
            magic: [b'n', b'+', b'1', b'\0'],
            ..Default::default()
        };

        write_affine(&mut hdr, &aff);
        Ok(PyNifti1Image { header: hdr, data })
    }

    #[staticmethod]
    fn from_array(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype_s = arr.getattr("dtype")?.str()?.to_string();
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("expected 3D array"));
        }
        let (nz, ny, nx) = (shape[0], shape[1], shape[2]);

        // Read data from numpy buffer, one copy only (Vec allocation).
        // Skips the intermediate Python bytes object that .tobytes() creates.
        let data = read_numpy_array(arr, &dtype_s, nz, ny, nx)?;

        // Default header (LPS identity)
        let (dt, bp) = dtype_bitpix(&data);
        let hdr = Nifti1Header {
            little_endian: true,
            dim: [3, nx as i32, ny as i32, nz as i32, 1, 0, 0, 0],
            datatype: dt,
            bitpix: bp,
            pixdim: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vox_offset: 352.0,
            srow_x: [-1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, -1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
            qform_code: 1,
            sform_code: 1,
            ..Default::default()
        };
        Ok(PyNifti1Image { header: hdr, data })
    }

    // ─── Setters ────────────────────────────────────────────────────────────

    fn set_affine(&mut self, affine: PyReadonlyArray2<f64>) {
        self.set_affine_from_nd(affine.as_array().to_owned());
    }

    fn set_spacing(&mut self, spacing: [f64; 3]) {
        assert!(spacing.iter().all(|&x| x > 0.0), "spacing must be > 0");
        let old = self.get_spacing();
        let mut aff = self.header.affine();
        for col in 0..3 {
            for row in 0..3 {
                aff[[row, col]] = aff[[row, col]] / old[col] * spacing[col];
            }
        }
        self.set_affine_from_nd(aff);
    }

    fn set_origin(&mut self, origin: [f64; 3]) {
        let mut aff = self.header.affine();
        aff[[0, 3]] = -origin[0];
        aff[[1, 3]] = -origin[1];
        aff[[2, 3]] = origin[2];
        self.set_affine_from_nd(aff);
    }

    fn set_direction(&mut self, direction: [[f64; 3]; 3]) {
        let d_ras = [
            -direction[0][0], -direction[0][1], -direction[0][2],
            -direction[1][0], -direction[1][1], -direction[1][2],
             direction[2][0],  direction[2][1],  direction[2][2],
        ];
        let sp = self.get_spacing();
        let mut aff = self.header.affine();
        for col in 0..3 {
            for row in 0..3 {
                aff[[row, col]] = d_ras[row * 3 + col] * sp[col];
            }
        }
        self.set_affine_from_nd(aff);
    }

    fn set_default_header(&mut self) {
        let aff = Array2::from_shape_vec((4, 4), vec![
            -1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]).unwrap();
        self.set_affine_from_nd(aff);
    }

    fn copy_information(&mut self, other: &PyNifti1Image) {
        self.set_affine_from_nd(other.header.affine());
    }

    // ─── Coordinate transforms ──────────────────────────────────────────────

    fn ijk2xyz(&self, ijk: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        let aff = self.header.affine();
        ijk.into_iter()
            .map(|[i, j, k]| {
                let ras_x = aff[[0,0]]*k + aff[[0,1]]*j + aff[[0,2]]*i + aff[[0,3]];
                let ras_y = aff[[1,0]]*k + aff[[1,1]]*j + aff[[1,2]]*i + aff[[1,3]];
                let ras_z = aff[[2,0]]*k + aff[[2,1]]*j + aff[[2,2]]*i + aff[[2,3]];
                [-ras_x, -ras_y, ras_z]
            })
            .collect()
    }

    fn xyz2ijk(&self, xyz: Vec<[f64; 3]>) -> Vec<[i32; 3]> {
        let aff = self.header.affine();
        let r = aff.slice(s![..3, ..3]);
        let inv = crate::mat3_inv(r);
        xyz.into_iter()
            .map(|[lps_x, lps_y, lps_z]| {
                let v = [-lps_x, -lps_y, lps_z];
                let ras = [v[0] - aff[[0,3]], v[1] - aff[[1,3]], v[2] - aff[[2,3]]];
                let x = inv[[0,0]]*ras[0] + inv[[0,1]]*ras[1] + inv[[0,2]]*ras[2];
                let y = inv[[1,0]]*ras[0] + inv[[1,1]]*ras[1] + inv[[1,2]]*ras[2];
                let z = inv[[2,0]]*ras[0] + inv[[2,1]]*ras[1] + inv[[2,2]]*ras[2];
                [z.round() as i32, y.round() as i32, x.round() as i32]
            })
            .collect()
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

fn arr_shape(data: &ImageData) -> &[usize] {
    match data {
        ImageData::F32(a) => a.shape(),
        ImageData::F64(a) => a.shape(),
        ImageData::U8(a)  => a.shape(),
        ImageData::U16(a) => a.shape(),
        ImageData::U32(a) => a.shape(),
        ImageData::U64(a) => a.shape(),
        ImageData::I8(a)  => a.shape(),
        ImageData::I16(a) => a.shape(),
        ImageData::I32(a) => a.shape(),
        ImageData::I64(a) => a.shape(),
    }
}

/// Write an affine matrix into a Nifti1Header (updates sform + pixdim).
fn write_affine(hdr: &mut Nifti1Header, aff: &Array2<f64>) {
    assert_eq!(aff.shape(), &[4, 4]);
    for i in 0..4 {
        hdr.srow_x[i] = aff[[0, i]];
        hdr.srow_y[i] = aff[[1, i]];
        hdr.srow_z[i] = aff[[2, i]];
    }
    hdr.sform_code = 1;
    let a = aff.slice(s![..3, ..3]);
    hdr.pixdim[1] = (a[[0,0]].powi(2)+a[[1,0]].powi(2)+a[[2,0]].powi(2)).sqrt();
    hdr.pixdim[2] = (a[[0,1]].powi(2)+a[[1,1]].powi(2)+a[[2,1]].powi(2)).sqrt();
    hdr.pixdim[3] = (a[[0,2]].powi(2)+a[[1,2]].powi(2)+a[[2,2]].powi(2)).sqrt();
}

impl PyNifti1Image {
    fn set_affine_from_nd(&mut self, aff: Array2<f64>) {
        write_affine(&mut self.header, &aff);
    }

    fn to_bytes(&self) -> Vec<u8> {
        use crate::header::*;
        let hdr = &self.header;
        let data_bytes = write_data(&self.data);

        let vox_off = 352u32;
        let total = vox_off as usize + data_bytes.len();
        let mut buf = vec![0u8; total];

        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        for i in 0..8 {
            buf[OFF_DIM + i*2 .. OFF_DIM + i*2 + 2]
                .copy_from_slice(&(hdr.dim[i] as i16).to_le_bytes());
        }

        buf[OFF_DATATYPE..OFF_DATATYPE+2].copy_from_slice(&hdr.datatype.to_le_bytes());
        buf[OFF_BITPIX..OFF_BITPIX+2].copy_from_slice(&hdr.bitpix.to_le_bytes());

        for i in 0..8 {
            buf[OFF_PIXDIM + i*4 .. OFF_PIXDIM + i*4 + 4]
                .copy_from_slice(&(hdr.pixdim[i] as f32).to_le_bytes());
        }

        buf[OFF_VOX_OFFSET..OFF_VOX_OFFSET+4]
            .copy_from_slice(&(vox_off as f32).to_le_bytes());

        for i in 0..4 {
            let o = OFF_SROW_X + i*4;
            buf[o..o+4].copy_from_slice(&(hdr.srow_x[i] as f32).to_le_bytes());
            let o = OFF_SROW_Y + i*4;
            buf[o..o+4].copy_from_slice(&(hdr.srow_y[i] as f32).to_le_bytes());
            let o = OFF_SROW_Z + i*4;
            buf[o..o+4].copy_from_slice(&(hdr.srow_z[i] as f32).to_le_bytes());
        }

        buf[OFF_SFORM_CODE..OFF_SFORM_CODE+2]
            .copy_from_slice(&(hdr.sform_code as i16).to_le_bytes());
        buf[OFF_QFORM_CODE..OFF_QFORM_CODE+2]
            .copy_from_slice(&(hdr.qform_code as i16).to_le_bytes());

        buf[OFF_QUATERN_B..OFF_QUATERN_B+4]
            .copy_from_slice(&(hdr.quatern_b as f32).to_le_bytes());
        buf[OFF_QUATERN_C..OFF_QUATERN_C+4]
            .copy_from_slice(&(hdr.quatern_c as f32).to_le_bytes());
        buf[OFF_QUATERN_D..OFF_QUATERN_D+4]
            .copy_from_slice(&(hdr.quatern_d as f32).to_le_bytes());
        buf[OFF_QOFFSET_X..OFF_QOFFSET_X+4]
            .copy_from_slice(&(hdr.qoffset_x as f32).to_le_bytes());
        buf[OFF_QOFFSET_Y..OFF_QOFFSET_Y+4]
            .copy_from_slice(&(hdr.qoffset_y as f32).to_le_bytes());
        buf[OFF_QOFFSET_Z..OFF_QOFFSET_Z+4]
            .copy_from_slice(&(hdr.qoffset_z as f32).to_le_bytes());

        buf[OFF_MAGIC..OFF_MAGIC+4].copy_from_slice(b"n+1\0");

        buf[vox_off as usize..][..data_bytes.len()].copy_from_slice(&data_bytes);
        buf
    }
}

// ─── Module registration ───────────────────────────────────────────────────

#[pymodule]
fn _nii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNifti1Image>()?;
    Ok(())
}
