"""
nii-rs: Pure Rust NIfTI-1 reader/writer.

The ``Nifti1Image`` class is implemented in Rust and imported directly.
This module re-exports it along with convenience aliases.
"""

from nii._nii import Nifti1Image

__all__ = ["Nifti1Image"]

#: Read an image from disk.  Alias for ``Nifti1Image.read()``.
#: The voxel data type is auto-detected from the file header.
read_image = Nifti1Image.read
