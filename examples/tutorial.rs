use nii;
use ndarray::{Array3, Array2};

fn main () {
    let pth = r"test_data\test.nii.gz";
    
    // Read Image, needs to specific type, eg: f32, u8, ...
    let im = nii::read_image::<f32>(pth);

    // get attrs, style same as like ITK
    let spacing: [f32; 3] = im.get_spacing();
    let origin: [f32; 3] = im.get_origin();
    let direction: [[f32; 3]; 3] = im.get_direction();
    let size: [u16; 3] = im.get_size();
    println!("spacing: {:?}, origin: {:?}, direction: {:?}, size: {:?}", spacing, origin, direction, size);

    // or print directly
    println!("{:?}", im);

    // get array, style same as ITK, i.e.: [z, y, x]
    let arr: &Array3<f32> = im.ndarray();
    println!("{:?}", arr);

    // get ownership
    let arr: Array3<f32> = im.into_ndarray();
    println!("{:?}", arr);

    // set attrs, style same as ITK; 
    // let im as **mut**
    let mut im = nii::read_image::<f32>(pth);
    im.set_origin([0.0, 1.0, 2.0]);
    im.set_spacing([1.0, 2.0, 3.0]);
    im.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    println!("{:?}", im);

    // or copy information from another image
    let im2 = nii::read_image::<f32>(pth);
    im.copy_infomation(&im2);

    // write image
    let pth = r"test_data\result.nii.gz";
    nii::write_image(&im, pth);

    // get affine, if you are more familiar with affine matrix
    // nibabel style
    let affine: Array2<f64> = im.get_affine();
    im.set_affine(affine);

    // make new image, based on ndarray + affine
    // nibabel style
    let new_affine: Array2<f64> = im.get_affine();
    let new_arr: Array3<f32> = im.ndarray().clone();
    let new_im = nii::new(new_arr, new_affine);
    println!("{:?}", new_im);

    // or simpleitk style
    let new_arr: Array3<f32> = im.ndarray().clone();
    let mut new_im = nii::get_image_from_array(new_arr);
    new_im.copy_infomation(&im);
    println!("{:?}", new_im);

    // That's all

}