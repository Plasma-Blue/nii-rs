import nii
import numpy as np

pth = rf"test_data\test.nii.gz"

# Read image — dtype auto-detected from file header
im = nii.Nifti1Image.read(pth)

# get attrs, style same as like ITK
spacing = im.get_spacing()
origin = im.get_origin()
direction = im.get_direction()
size = im.get_size()
print(f"spacing: {spacing}, origin: {origin}, direction: {direction}, size: {size}")

# or print directly (uses __str__)
print(im)

# get array, style same as ITK, i.e.: [z, y, x]
arr = im.ndarray()
print(arr)

# set attrs, style same as ITK
im2 = nii.Nifti1Image.read(pth)
im2.set_origin([0.0, 1.0, 2.0])
im2.set_spacing([1.0, 2.0, 3.0])
im2.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
print(im2)

# write image
pth = rf"test_data\result.nii.gz"
im2.write(pth)

# get affine, nibabel style
affine = im.get_affine()
im.set_affine(affine)

# copy information from another image
im3 = nii.Nifti1Image.read(pth)
im.copy_information(im3)

print("Done!")
