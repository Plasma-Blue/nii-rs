//! Rust library for reading/writing NIfTI-1 (nii.gz) files, with SimpleITK/NiBabel-like APIs, native Rust support, and Python bindings for cross-language performance.
//!
//! If you have used SimpleITK/NiBabel, you will definitely love this and get started right away! 🕶
//!
//! ## 🎨Features
//!
//! - 🚀**Pure Rust Implementation**: Thanks to [nifti-rs](https://github.com/Enet4/nifti-rs), I/O speed is comparable to SimpleITK and slightly faster than NiBabel.
//!
//! - ✨ **Carefully Designed API**: *Super easy to use*, with no learning curve; enjoy a consistent experience in Rust as in Python. Ideal for developers familiar with ITK-style libraries.
//!
//! - 🛠️**Rust-Python bindings**: Write heavy operations in Rust and easily call them in Python, combining the performance of Rust with the flexibility of Python.
//!
//! ## 🔨Install
//!
//! For Rust projects, add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! nii-rs = "*"
//! ```
//! For Python, install via pip:
//!
//! ```sh
//! pip install nii-rs
//! ```
//!
//! ## 🥒Develop
//!
//! To start developing with nii-rs, use the following command:
//!
//! `maturin dev -r`
//!
//! ## 📘Examples
//!
//! For details, please refer to the [rust examples](examples/tutorial.rs) and [python examples](examples/tutorial.py)。
//!
//! ### 🦀Rust
//!
//! ```rust
//! use nii;
//!
//! // read image
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
//! // get array, style same as SimpleITK, i.e.: [z, y, x]
//! let arr: &Array3<f32> = im.ndarray();
//!
//! // write image
//! nii::write_image(&im, "result.nii.gz")
//! ```
//!
//! ## 🔒License
//!
//! Apache License 2.0 & MIT

mod bind;
mod image;
mod utils;

pub use image::*;
