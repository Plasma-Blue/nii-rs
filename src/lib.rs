use pyo3::prelude::*;
pub mod image;
pub mod bind;
pub mod utils;

pub use bind::Nifti1ImageF32;


#[pyfunction]
pub fn read_fimage(path: &str) -> Nifti1ImageF32 {
    Nifti1ImageF32::read(path).unwrap()
}

/// A Python module implemented in Rust.
#[pymodule]
fn _niirs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Nifti1ImageF32>()?;
    m.add_function(wrap_pyfunction!(read_fimage, m)?)?;
    Ok(())
}
