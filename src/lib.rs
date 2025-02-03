//! Rust library for reading/writing NIfTI-1 (nii.gz) files, with SimpleITK/Nibabel-like APIs, native Rust support, and Python bindings for cross-language performance.
//!
//! If you have used SimpleITK/Nibabel, you will definitely love this and get started right away! 🕶
//!
//! ## 🎨Features
//!
//! - 🚀**Pure Rust Implementation**: I/O speed is comparable to SimpleITK and slightly faster than nibabel.
//!
//! - ✨ **Carefully Designed API**: *Super easy to use*, with no learning curve; enjoy a consistent experience in Rust as in Python.
//!
//! - 🛠️**Rust-Python bindings**: you can write heavy operations in Rust and easily call them in Python.
//!
//! ## 🔨Install
//!
//! `cargo add nii-rs` for rust project and `pip install niirs` for python.
//!
//! ## 🥒Develop
//!
//! `maturin dev`
//!
//! ## 📘Examples
//!
//! For details, please refer to the [rust examples](https://github.com/Plasma-Blue/nii-rs/tree/master/examples/tutorial.rs) and [python examples](https://github.com/Plasma-Blue/nii-rs/tree/master/examples/tutorial.py)。
//!
//! ### Rust
//!
//! ```rust
//! use nii;
//!
//! let im = nii::read_image::<f32>("test.nii.gz");
//!
//! // get attrs, style same as SimpleITK
//! let spacing: [f32; 3] = im.get_spacing();
//! let origin: [f32; 3] = im.get_origin();
//! let direction: [[f32; 3]; 3] = im.get_direction();
//!
//! // get affine, style same as nibabel
//! let affine = im.get_affine();
//!
//! // get array, style same as SimpleITK, i.e.: [z, y, z]
//! let arr: &Array3<f32> = im.ndarray();
//!
//! // write image
//! nii::write_image(arr, "result.nii.gz")
//! ```
//!
//! ### Python
//! ```python
//! import niirs
//!
//! im = niirs.read_image("test.nii.gz")
//!
//! # get attrs, style same as like SimpleITK
//! spacing = im.get_spacing()
//! origin = im.get_origin()
//! direction = im.get_direction()
//!
//! # get affine, style same as nibabel
//! affine = im.get_affine()
//!
//! # get array, style same as SimpleITK, i.e.: [z, y, x]
//! arr = im.ndarray()
//!
//! # write image
//! niirs.write_image(arr, "result.nii.gz")
//! ```
//!
//! ## 🔒License
//!
//! Apache License 2.0 & MIT

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
