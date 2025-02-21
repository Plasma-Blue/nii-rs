from typing import List, Union

import numpy as np
from typing_extensions import Self

from nii import _nii

__all__ = ["read_image", "write_image", "new", "get_image_from_array"]


class Nifti1Image:
    """
    Python wrapper for rust's Nifti1Image<T> binding.
    """

    def __init__(self, Nifti1ImageT, dtype: np.dtype):
        self._rs = Nifti1ImageT
        self.dtype = dtype

    def __str__(self):
        info = f"Size: {self.get_size()}\n"
        info += f"Spacing: {self.get_spacing()}\n"
        info += f"Origin: {self.get_origin()}\n"
        info += f"Direction: {self.get_direction()}"
        return info

    def get_spacing(self) -> List[float]:
        """
        Return Spacing (ITK style, i.e.: [x, y, z])
        """
        return self._rs.get_spacing()

    def get_size(self) -> List[int]:
        """
        Return Size (ITK style, i.e.: [x, y, z])
        """
        return self._rs.get_size()

    def get_origin(self) -> List[float]:
        """
        Return Origin (ITK style, i.e.: [x, y, z])
        """
        return self._rs.get_origin()

    def get_direction(self) -> List[list]:
        """
        Return Direction (ITK style, 3x3 list, i.e.: [[a,b,c], [d,e,f], [g,h,i]])
        """
        return self._rs.get_direction()

    def get_unit_size(self) -> float:
        """
        Return unit voxel size (mm3). Very useful when calc volumes of label.
        """
        return self._rs.get_unit_size()

    def get_affine(self) -> np.ndarray:
        """
        Set affine matrix (4x4, nibabel style).
        """
        return self._rs.get_affine()

    def set_spacing(self, spacing: Union[list, np.ndarray]):
        """
        Set Spacing for Nifti1Image. (ITK style, i.e.: [x, y, z])
        """
        assert len(spacing) == 3
        if isinstance(spacing, np.ndarray):
            spacing = spacing.tolist()

        self._rs.set_spacing(spacing)

    def set_origin(self, origin: Union[list, np.ndarray]):
        """
        Set Origin for Nifti1Image. (ITK style, i.e.: [x, y, z])
        """
        assert len(origin) == 3
        if isinstance(origin, np.ndarray):
            origin = origin.tolist()

        self._rs.set_origin(origin)

    def set_direction(self, direction: Union[list, np.ndarray]):
        """
        Set Direction for Nifti1Image. (ITK style, 3x3 list, i.e.: [[a,b,c], [d,e,f], [g,h,i]])
        """
        assert len(direction) == 3
        if isinstance(direction, np.ndarray):
            direction = direction.tolist()

        self._rs.set_direction(direction)

    def copy_infomation(self, im: Self):
        """
        Copy informations.
        """
        self.set_affine(im.get_affine())

    def ijk2xyz(self, ijk: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """
        Pixel indices i,j,k -> Physical positions (ITK style, i.e.: [x, y, z])
        No restriction on whether ijk or xyz are within the shape
        """
        is_list = True if isinstance(ijk, list) else False
        ijk = np.array(ijk)
        assert ijk.ndim in (1, 2), f"ijk ndim = {ijk.ndim}, ndim mismatch"
        if ijk.ndim == 1:
            ijk = ijk[None, ...]
        xyz = np.array(self._rs.ijk2xyz(ijk)).squeeze()
        if is_list:
            xyz = xyz.tolist()
        return xyz


    def xyz2ijk(self, xyz: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """
        Physical positions -> Pixel indices i,j,k (ITK style, i.e.: [x, y, z])
        No restriction on whether xyz or ijk are within the shape, no restriction on ijk being positive, please be careful
        """
        is_list = True if isinstance(xyz, list) else False
        xyz = np.array(xyz)
        assert xyz.ndim in (1, 2), f"ijk ndim = {xyz.ndim}, ndim mismatch"
        if xyz.ndim == 1:
            xyz = xyz[None, ...]
        ijk = np.array(self._rs.xyz2ijk(xyz)).squeeze()
        if is_list:
            ijk = ijk.tolist()
        return ijk
    
    def set_default_header(self):
        """
        Equals:
        ```python
        im.set_spacing([1,1,1])
        im.set_origin([0,0,0])
        im.set_direction([[1,0,0],[0,1,0],[0,0,1]])
        ```
        """
        self._rs.set_default_header()

    def set_affine(self, affine: Union[list, np.ndarray]):
        """
        Set Affine for Nifti1Image. (nibabel style, i.e.: 4x4 list/ndarray)
        """
        assert len(affine) == 4
        affine = np.array(affine).astype(np.float64)

        self._rs.set_affine(affine)

    def ndarray(self) -> np.ndarray:
        """
        Get Array from Image. (ITK style, i.e.: [z, y, x])
        """
        return self._rs.ndarray()

    def to_rs(self):
        '''
        Expose the Rust Nifti1Image<T> object for interaction with Rust custom functions.
        '''
        return self._rs


def read_image(pth: str, dtype: np.dtype = np.float32) -> Nifti1Image:
    """
    Read image from disk.
    """
    funcs = {
        np.float32: _nii.read_image_f32,
        np.float64: _nii.read_image_f64,
        np.int8: _nii.read_image_i8,
        np.int16: _nii.read_image_i16,
        np.int32: _nii.read_image_i32,
        np.int64: _nii.read_image_i64,
        np.uint8: _nii.read_image_u8,
        np.uint16: _nii.read_image_u16,
        np.uint32: _nii.read_image_u32,
        np.uint64: _nii.read_image_u64,
    }
    return Nifti1Image(funcs.get(dtype, np.float32)(pth), dtype)


def write_image(im: Nifti1Image, pth: str):
    """
    Write image to disk.
    """
    dtype = im.dtype
    funcs = {
        np.float32: _nii.write_image_f32,
        np.float64: _nii.write_image_f64,
        np.int8: _nii.write_image_i8,
        np.int16: _nii.write_image_i16,
        np.int32: _nii.write_image_i32,
        np.int64: _nii.write_image_i64,
        np.uint8: _nii.write_image_u8,
        np.uint16: _nii.write_image_u16,
        np.uint32: _nii.write_image_u32,
        np.uint64: _nii.write_image_u64,
    }
    return funcs.get(dtype, np.float32)(im._rs, pth)


def new(arr: np.ndarray, affine: np.ndarray) -> Nifti1Image:
    '''
    Make a new Nifti1Image using array and affine like nibabel.
    '''
    dtype = arr.dtype.type
    affine = affine.astype(np.float64)
    funcs = {
        np.float32: _nii.new_f32,
        np.float64: _nii.new_f64,
        np.int8: _nii.new_i8,
        np.int16: _nii.new_i16,
        np.int32: _nii.new_i32,
        np.int64: _nii.new_i64,
        np.uint8: _nii.new_u8,
        np.uint16: _nii.new_u16,
        np.uint32: _nii.new_u32,
        np.uint64: _nii.new_u64,
    }
    return Nifti1Image(funcs.get(dtype, np.float32)(arr, affine), dtype)


def get_image_from_array(arr: np.ndarray) -> Nifti1Image:
    '''
    Get image from array with default header.
    '''
    dtype = arr.dtype.type
    funcs = {
        np.float32: _nii.get_image_from_array_f32,
        np.float64: _nii.get_image_from_array_f64,
        np.int8: _nii.get_image_from_array_i8,
        np.int16: _nii.get_image_from_array_i16,
        np.int32: _nii.get_image_from_array_i32,
        np.int64: _nii.get_image_from_array_i64,
        np.uint8: _nii.get_image_from_array_u8,
        np.uint16: _nii.get_image_from_array_u16,
        np.uint32: _nii.get_image_from_array_u32,
        np.uint64: _nii.get_image_from_array_u64,
    }
    return Nifti1Image(funcs.get(dtype, np.float32)(arr), dtype)
