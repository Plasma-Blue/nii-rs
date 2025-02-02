use ndarray::Array3;
use nifti::
    header::NiftiHeader
;
use pyo3::prelude::*;
use crate::image::Nifti1Image;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyArray3};
use pyo3::{Bound, PyResult, Python};


#[derive(Clone)]
#[pyclass]
pub struct Nifti1ImageF32 {
    inner: Nifti1Image<f32>,
}

pub trait Nifti1ImageTrait1 {
    fn header(&self) -> &NiftiHeader;
    fn header_mut(&mut self) -> &mut NiftiHeader;
    fn ndarray(&self) -> &Array3<f32>;
}

#[pymethods]
impl Nifti1ImageF32
{
    #[staticmethod]
    pub fn read(path: &str) -> PyResult<Self>
    {
        let inner = Nifti1Image::<f32>::read(path);
        Ok(Nifti1ImageF32{inner})
    }

    pub fn get_spacing(&self) -> [f32; 3] {
        self.inner.get_spacing()
    }

    pub fn get_size(&self) -> [u16; 3] {
        self.inner.get_size()
    }

    pub fn get_origin(&self) -> [f32; 3] {
        self.inner.get_origin()
    }

    pub fn get_direction(&self) -> [[f32; 3]; 3] {
        self.inner.get_direction()
    }

    pub fn get_unit_size(&self) -> f32 {
        self.inner.get_unit_size()
    }

    pub fn write(&self, path: &str) -> () {
        self.inner.write(path);
    }

    pub fn set_spacing(&mut self, spacing: [f32; 3]) {
        self.inner.set_spacing(spacing);
    }

    pub fn set_origin(&mut self, origin: [f32; 3]) {
        self.inner.set_origin(origin);
    }

    pub fn set_direction(&mut self, direction: [[f32; 3]; 3]) {
        self.inner.set_direction(direction);
    }

    pub fn copy_infomation(&mut self, im: &Nifti1ImageF32) {
        self.inner.copy_infomation(&im.inner);
    }

    pub fn set_default_header(&mut self) {
        self.inner.set_default_header();
    }

    pub fn set_affine(
        &mut self,
        affine: PyReadonlyArray2<f64>,
    ) //-> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let affine = affine.as_array().to_owned();
        self.inner.set_affine(affine);
    }

    pub fn get_affine<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let y = self.inner.get_affine();
        Ok(y.into_pyarray(py))
    }

    pub fn ndarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let y = self.inner.ndarray().clone();
        Ok(y.into_pyarray(py))
    }
}
