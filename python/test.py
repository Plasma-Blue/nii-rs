import _niirs as niirs
import nibabel as nib
import time

import numpy as np
import SimpleITK as sitk

t = time.time()
im = niirs.read_fimage(rf"test.nii.gz")
arr = im.ndarray()
print(time.time() - t)

t = time.time()
im2 = nib.load(rf"test.nii.gz").get_fdata()
print(time.time() - t)

t = time.time()
arr3 = sitk.GetArrayFromImage(sitk.ReadImage(rf"test.nii.gz"))
print(time.time() - t)


np.testing.assert_equal(arr, arr3)