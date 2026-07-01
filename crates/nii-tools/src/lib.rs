//! nii-tools: example downstream crate that accepts ``Nifti1Image`` from Python.
//!
//! Demonstrates cross-crate object passing via Python's dynamic dispatch.
//! Since PyO3 cannot match ``#[pyclass]`` types across independent extension
//! modules, we accept ``Bound<'_, PyAny>`` and call methods through Python.

use pyo3::prelude::*;

/// Print image info by calling Python methods on a ``Nifti1Image`` object.
#[pyfunction]
fn describe(im: &Bound<'_, PyAny>) -> PyResult<String> {
    let size: Vec<u32> = im.call_method0("get_size")?.extract()?;
    let spacing: Vec<f64> = im.call_method0("get_spacing")?.extract()?;
    Ok(format!("Image(size={size:?}, spacing={spacing:?})"))
}

/// Compute the unit voxel volume in mm³.
#[pyfunction]
fn voxel_volume(im: &Bound<'_, PyAny>) -> PyResult<f64> {
    im.call_method0("get_unit_size")?.extract()
}

/// Check if two images have the same spatial metadata.
#[pyfunction]
fn same_space(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let sz_a: Vec<u32> = a.call_method0("get_size")?.extract()?;
    let sz_b: Vec<u32> = b.call_method0("get_size")?.extract()?;
    let sp_a: Vec<f64> = a.call_method0("get_spacing")?.extract()?;
    let sp_b: Vec<f64> = b.call_method0("get_spacing")?.extract()?;
    Ok(sz_a == sz_b && sp_a == sp_b)
}

#[pymodule]
fn nii_tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_volume, m)?)?;
    m.add_function(wrap_pyfunction!(same_space, m)?)?;
    Ok(())
}
