use pyo3::prelude::*;
pub mod bind;
pub mod image;
pub mod utils;

pub use bind::*;
pub use image::*;
use paste::paste;

macro_rules! bind_py_wrapper {
    ($type_name:ident, $py_struct:ident, $m:ident) => {
        paste! {
            $m.add_class::<$py_struct>()?;
            $m.add_function(wrap_pyfunction!([<read_image_$type_name>], $m)?)?;
            $m.add_function(wrap_pyfunction!([<write_image_$type_name>], $m)?)?;
            $m.add_function(wrap_pyfunction!([<new_$type_name>], $m)?)?;
            $m.add_function(wrap_pyfunction!([<get_image_from_array_$type_name>], $m)?)?;
        }
    };
}

/// A Python module implemented in Rust.
#[pymodule]
fn _niirs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bind_py_wrapper!(f32, Nifti1ImageF32, m);
    bind_py_wrapper!(f64, Nifti1ImageF64, m);
    bind_py_wrapper!(u8, Nifti1ImageU8, m);
    bind_py_wrapper!(u16, Nifti1ImageU16, m);
    bind_py_wrapper!(u32, Nifti1ImageU32, m);
    bind_py_wrapper!(u64, Nifti1ImageU64, m);
    bind_py_wrapper!(i8, Nifti1ImageI8, m);
    bind_py_wrapper!(i16, Nifti1ImageI16, m);
    bind_py_wrapper!(i32, Nifti1ImageI32, m);
    bind_py_wrapper!(i64, Nifti1ImageI64, m);
    Ok(())
}
