//! # nii-rs
//!
//! Pure Rust NIfTI-1 medical image reader/writer with SimpleITK/NiBabel-like API.
//!
//! ```rust
//! use nii;
//!
//! // Read a NIfTI-1 image
//! let im = nii::read_image::<f32>("test.nii.gz").unwrap();
//!
//! // ITK-style accessors
//! let spacing: [f64; 3] = im.get_spacing();
//! let origin: [f64; 3] = im.get_origin();
//! let direction: [[f64; 3]; 3] = im.get_direction();
//! let size: [u32; 3] = im.get_size();
//!
//! // Nibabel-style affine
//! let affine = im.get_affine();
//!
//! // Voxel data (ITK style: [z, y, x])
//! let arr = im.ndarray();
//!
//! // Write
//! nii::write_image(&im, "result.nii.gz").unwrap();
//! ```

mod header;
mod image;
#[cfg(feature = "python")]
mod bind;

#[cfg(feature = "python")]
pub use bind::PyNifti1Image;

pub use header::{Nifti1Header, NiftiError, dtype};
pub use image::*;
