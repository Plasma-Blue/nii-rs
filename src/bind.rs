//! This module defines rust-python bind.

use crate::{get_image_from_array, new, Nifti1Image};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use paste::paste;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};

/// Since pyo3 does not support generic classes, we generate specific classes for various types through macros to avoid repetitive code.
///
/// To avoid explicit Python interface, we rewrapped the classes in Python, making it look like a sandwich structure.
///
/// In fact, according to the discussion [here](https://github.com/nipy/nibabel/issues/1046), the commonly used types for nii.gz are only u8, i16, and f32. Others are not even standard NIfTI types. Regardless, we have provided support for them.
macro_rules! impl_py_wrapper {
    ($type:ty, $py_struct:ident) => {
        #[pyclass]
        #[derive(Clone)]
        pub struct $py_struct {
            inner: Nifti1Image<$type>,
        }

        #[pymethods]
        impl $py_struct {
            #[staticmethod]
            pub fn read(path: &str) -> PyResult<Self> {
                let inner = Nifti1Image::<$type>::read(path);
                Ok($py_struct { inner })
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

            pub fn copy_infomation(&mut self, im: &$py_struct) {
                self.inner.copy_infomation(&im.inner);
            }

            pub fn set_default_header(&mut self) {
                self.inner.set_default_header();
            }

            pub fn set_affine(&mut self, affine: PyReadonlyArray2<f64>) {
                let affine = affine.as_array().to_owned();
                self.inner.set_affine(affine);
            }

            pub fn get_affine<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let y = self.inner.get_affine();
                Ok(y.into_pyarray(py))
            }

            pub fn ndarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<$type>>> {
                let y = self.inner.ndarray().clone();
                Ok(y.into_pyarray(py))
            }
        }
    };
}

macro_rules! function_py_wrapper {
    ($type:ty, $func_name:ident, $py_struct:ident) => {
        paste! {
            #[pyfunction]
            pub fn [<read_image_$func_name>](path: &str) -> $py_struct {
                $py_struct::read(path).unwrap()
            }

            #[pyfunction]
            pub fn [<write_image_$func_name>](im: $py_struct, path: &str) -> ()
            {
                im.write(path);
            }

            #[pyfunction]
            pub fn [<new_$func_name>](
                _py: Python<'_>,
                ndarray: PyReadonlyArray3<$type>,
                affine: PyReadonlyArray2<f64>
            ) -> $py_struct
            {
                let ndarray = ndarray.as_array().to_owned();
                let affine = affine.as_array().to_owned();
                $py_struct {
                    inner: new(ndarray, affine),
                }
            }

            #[pyfunction]
            pub fn [<get_image_from_array_$func_name>](
                _py: Python<'_>,
                ndarray: PyReadonlyArray3<$type>,
            ) -> $py_struct
            {
                let ndarray = ndarray.as_array().to_owned();
                $py_struct {
                    inner: get_image_from_array::<$type>(ndarray),
                }
            }
        }
    };
}

impl_py_wrapper!(f32, Nifti1ImageF32);
impl_py_wrapper!(f64, Nifti1ImageF64);
impl_py_wrapper!(u8, Nifti1ImageU8);
impl_py_wrapper!(u16, Nifti1ImageU16);
impl_py_wrapper!(u32, Nifti1ImageU32);
impl_py_wrapper!(u64, Nifti1ImageU64);
impl_py_wrapper!(i8, Nifti1ImageI8);
impl_py_wrapper!(i16, Nifti1ImageI16);
impl_py_wrapper!(i32, Nifti1ImageI32);
impl_py_wrapper!(i64, Nifti1ImageI64);

function_py_wrapper!(f32, f32, Nifti1ImageF32);
function_py_wrapper!(f64, f64, Nifti1ImageF64);
function_py_wrapper!(u8, u8, Nifti1ImageU8);
function_py_wrapper!(u16, u16, Nifti1ImageU16);
function_py_wrapper!(u32, u32, Nifti1ImageU32);
function_py_wrapper!(u64, u64, Nifti1ImageU64);
function_py_wrapper!(i8, i8, Nifti1ImageI8);
function_py_wrapper!(i16, i16, Nifti1ImageI16);
function_py_wrapper!(i32, i32, Nifti1ImageI32);
function_py_wrapper!(i64, i64, Nifti1ImageI64);
