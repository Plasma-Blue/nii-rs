import nii
import numpy as np

pth = rf"test_data\test.nii.gz"

im = nii.read_image(pth, dtype=np.float32)

# get attrs, style same as like ITK
spacing = im.get_spacing()
origin = im.get_origin()
direction = im.get_direction()
size = im.get_size()
print(f"spacing: {spacing}, origin: {origin}, direction: {direction}, size: {size}")

# or print directly
print(im)

# get array, style same as ITK, i.e.: [z, y, x]
arr = im.ndarray()
print(arr)

# set attrs, style same as ITK; 
im = nii.read_image(pth, dtype=np.float32)
im.set_origin([0.0, 1.0, 2.0])
im.set_spacing([1.0, 2.0, 3.0])
im.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
print(im)

# or copy information from another image
im2 = nii.read_image(pth)
im.copy_infomation(im2)

# write image
pth = rf"test_data\result.nii.gz"
nii.write_image(im, pth)

# get affine, if you are more familiar with affine matrix
# nibabel style
affine = im.get_affine()
im.set_affine(affine)

# make new image, based on ndarray + affine
# nibabel style
new_affine = im.get_affine()
new_arr = im.ndarray()
new_im = nii.new(new_arr, new_affine)
print(new_im)

# or simpleitk style
new_arr = im.ndarray()
new_im = nii.get_image_from_array(new_arr)
new_im.copy_infomation(im)
print(new_im)

# that's all
