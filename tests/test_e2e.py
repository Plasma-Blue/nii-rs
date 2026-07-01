"""
End-to-end test for nii-rs Python bindings.

Tests read/write round-trip using only Python.
Run with:  python tests/test_e2e.py
"""

import os
import tempfile

import numpy as np
from nii._nii import Nifti1Image


def test_roundtrip_f32():
    """Create f32 array → write .nii → read back → verify."""
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)  # [z, y, x]

    im = Nifti1Image.from_array(arr)

    assert im.get_size() == [4, 3, 2], f"size mismatch: {im.get_size()}"
    assert np.allclose(im.ndarray(), arr), "ndarray mismatch after from_array"

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        pth = f.name
    try:
        im.write(pth)
        im2 = Nifti1Image.read(pth)

        assert im2.get_size() == [4, 3, 2], f"read size mismatch: {im2.get_size()}"
        assert np.allclose(im2.ndarray(), arr), "read-back data mismatch"
        print(f"[PASS] f32 round-trip ({pth})")
    finally:
        os.unlink(pth)


def test_roundtrip_u8():
    """Same for uint8."""
    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    im = Nifti1Image.from_array(arr)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        pth = f.name
    try:
        im.write(pth)
        im2 = Nifti1Image.read(pth)
        assert np.array_equal(im2.ndarray(), arr), "uint8 round-trip failed"
        print(f"[PASS] u8  round-trip gz ({pth})")
    finally:
        os.unlink(pth)


def test_affine_roundtrip():
    """from_array → get_affine → set_affine → write → read → verify."""
    arr = np.random.rand(4, 5, 6).astype(np.float32)
    im = Nifti1Image.from_array(arr)

    # Modify affine
    aff = im.get_affine()
    aff[0, 3] = 42.0  # set origin x
    im.set_affine(aff)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        pth = f.name
    try:
        im.write(pth)
        im2 = Nifti1Image.read(pth)
        aff2 = im2.get_affine()
        assert abs(aff2[0, 3] - 42.0) < 1e-6, f"origin x mismatch: {aff2[0, 3]}"
        assert np.allclose(im2.ndarray(), arr), "affine round-trip data mismatch"
        print(f"[PASS] affine round-trip ({pth})")
    finally:
        os.unlink(pth)


def test_spacing_origin_direction():
    """set_spacing / set_origin / set_direction → get_* round-trip."""
    arr = np.ones((4, 5, 6), dtype=np.float32)
    im = Nifti1Image.from_array(arr)

    im.set_spacing([2.0, 3.0, 4.0])
    im.set_origin([1.0, 2.0, 3.0])
    im.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    sp = im.get_spacing()
    assert all(abs(a - b) < 1e-6 for a, b in zip(sp, [2.0, 3.0, 4.0])), f"spacing: {sp}"
    ori = im.get_origin()
    assert all(abs(a - b) < 1e-6 for a, b in zip(ori, [1.0, 2.0, 3.0])), (
        f"origin: {ori}"
    )

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        pth = f.name
    try:
        im.write(pth)
        im2 = Nifti1Image.read(pth)
        sp2 = im2.get_spacing()
        ori2 = im2.get_origin()
        assert all(abs(a - b) < 1e-5 for a, b in zip(sp2, [2.0, 3.0, 4.0])), (
            f"spacing read: {sp2}"
        )
        assert all(abs(a - b) < 1e-5 for a, b in zip(ori2, [1.0, 2.0, 3.0])), (
            f"origin read: {ori2}"
        )
        print(f"[PASS] spacing/origin/direction round-trip")
    finally:
        os.unlink(pth)


def test_new_with_affine():
    """Nifti1Image.new(arr, affine) → read back."""
    arr = np.random.rand(3, 4, 5).astype(np.float32)
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = 2.0
    aff[1, 3] = -10.0

    im = Nifti1Image.new(arr, aff)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        pth = f.name
    try:
        im.write(pth)
        im2 = Nifti1Image.read(pth)
        aff2 = im2.get_affine()
        assert abs(aff2[0, 0] - 2.0) < 1e-6, f"sform[0,0] mismatch: {aff2[0, 0]}"
        assert np.allclose(im2.ndarray(), arr), "new-with-affine data mismatch"
        print(f"[PASS] new() with custom affine")
    finally:
        os.unlink(pth)


def test_str_repr():
    """__str__ produces readable output."""
    arr = np.zeros((2, 3, 4), dtype=np.float32)
    im = Nifti1Image.from_array(arr)
    s = str(im)
    assert "Size" in s and "Spacing" in s and "Origin" in s and "Direction" in s
    print(f"[PASS] __str__")


def test_copy_information():
    """copy_information between two images."""
    arr1 = np.random.rand(4, 5, 6).astype(np.float32)
    arr2 = np.ones((4, 5, 6), dtype=np.float32)
    im1 = Nifti1Image.from_array(arr1)
    im2 = Nifti1Image.from_array(arr2)

    im1.set_origin([10.0, 20.0, 30.0])
    im2.copy_information(im1)

    assert np.allclose(im2.get_origin(), [10.0, 20.0, 30.0])
    # data should be unchanged
    assert np.allclose(im2.ndarray(), arr2)
    print(f"[PASS] copy_information")


def test_ijk2xyz_xyz2ijk():
    """Coordinate transforms round-trip."""
    arr = np.ones((4, 5, 6), dtype=np.float32)
    im = Nifti1Image.from_array(arr)

    ijk = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    xyz = im.ijk2xyz(ijk)
    ijk2 = im.xyz2ijk(xyz)
    assert all(
        ijk2[i] == [int(ijk[i][0]), int(ijk[i][1]), int(ijk[i][2])]
        for i in range(len(ijk))
    ), f"ijk→xyz→ijk failed: {ijk} → {xyz} → {ijk2}"
    print(f"[PASS] ijk2xyz / xyz2ijk")


if __name__ == "__main__":
    test_roundtrip_f32()
    test_roundtrip_u8()
    test_affine_roundtrip()
    test_spacing_origin_direction()
    test_new_with_affine()
    test_str_repr()
    test_copy_information()
    test_ijk2xyz_xyz2ijk()
    print("\n✅ All tests passed!")
