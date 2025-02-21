//! This is the core library.

use crate::utils::*;
use bytemuck::Pod;
use ndarray::prelude::*;
use ndarray::{Array2, Array3};
use nifti::{
    header::NiftiHeader, writer::WriterOptions, DataElement, IntoNdArray, NiftiObject,
    ReaderOptions,
};
use rayon::prelude::*;
use std::fmt;
use std::path::Path;

/// Core struct of nii-rs.
/// Nifti1Image = header + array
#[derive(Clone)]
pub struct Nifti1Image<T> {
    pub header: NiftiHeader,
    pub ndarray: Array3<T>,
}

impl<T> Nifti1Image<T>
where
    T: DataElement + Pod,
{
    /// Read image and return `Nifti1Image<T>`. Rewrapped the API of nifti-rs.
    pub fn read(path: impl AsRef<Path>) -> Nifti1Image<T> {
        let path = path.as_ref();

        let im = ReaderOptions::new()
            .read_file(path)
            .expect("Failed to read NIfTI file");

        let header = im.header().clone();

        let ndarray = im
            .into_volume()
            .into_ndarray::<T>()
            .expect("msg")
            .into_dimensionality()
            .expect("msg");
        let ndarray = ndarray.permuted_axes((2, 1, 0)); // nifti-rs style -> ITK style

        Nifti1Image { header, ndarray }
    }

    /// Make a new `Nifti1Image<T>` if struct members are private.
    pub fn new(header: NiftiHeader, ndarray: Array3<T>) -> Self {
        Self { header, ndarray }
    }

    /// Get header from nifti-rs.
    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    /// Get mut header from nifti-rs.
    pub fn header_mut(&mut self) -> &mut NiftiHeader {
        &mut self.header
    }

    /// Return Spacing (ITK style, i.e.: [x, y, z])
    pub fn get_spacing(&self) -> [f32; 3] {
        let header: &NiftiHeader = self.header();
        [header.pixdim[1], header.pixdim[2], header.pixdim[3]]
    }

    /// Return Size (ITK style, i.e.: [x, y, z])
    pub fn get_size(&self) -> [u16; 3] {
        let ndarray = self.ndarray();
        let shape = ndarray.shape();
        [shape[2] as u16, shape[1] as u16, shape[0] as u16] // ITK style
    }

    /// Return Origin (ITK style, i.e.: [x, y, z])
    pub fn get_origin(&self) -> [f32; 3] {
        let header: &NiftiHeader = self.header();
        [-header.srow_x[3], -header.srow_y[3], header.srow_z[3]] // nifti-rs style -> ITK style
    }

    /// Return Direction (ITK style, 3x3 list, i.e.: \[\[a,b,c\], \[d,e,f\], \[g,h,i\]\])
    pub fn get_direction(&self) -> [[f32; 3]; 3] {
        let header: &NiftiHeader = self.header();
        [
            [-header.srow_x[0], -header.srow_x[1], -header.srow_x[2]],
            [-header.srow_y[0], -header.srow_y[1], -header.srow_y[2]],
            [header.srow_z[0], header.srow_z[1], header.srow_z[2]],
        ] // nifti-rs style -> ITK style
    }

    /// Return unit voxel size (mm3). Very useful when calc volumes of label.
    pub fn get_unit_size(&self) -> f32 {
        let spacing = self.get_spacing();
        spacing[0] * spacing[1] * spacing[2]
    }

    /// Get Array from Image. (ITK style, i.e.: [z, y, x])
    pub fn ndarray(&self) -> &Array3<T> {
        &self.ndarray
    }

    /// Get Array from Image with ownship. (ITK style, i.e.: [z, y, x])
    pub fn into_ndarray(self) -> Array3<T> {
        self.ndarray
    }

    /// Write Nifti1Image<T> to disk. Rewrapped the API of nifti-rs.
    pub fn write(&self, path: impl AsRef<Path>) -> () {
        let header = self.header();
        let data = self.ndarray().view().permuted_axes((2, 1, 0)); // ITK style -> nifti-rs style
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&data)
            .unwrap();
    }

    /// Return affine matrix (4x4, nibabel style).
    pub fn get_affine(&self) -> Array2<f64> {
        let na_arr = self.header().affine::<f64>().transpose(); // nifti-rs style -> nibabel style
        na2nd_4x4(na_arr)
    }

    /// Set affine matrix (4x4, nibabel style).
    pub fn set_affine(&mut self, affine: Array2<f64>) {
        let affine = nd2na_4x4(affine);
        self.header_mut().set_affine::<f64>(&affine.transpose()); // nibabel style -> nifti-rs style
    }

    /// Set Spacing for Nifti1Image. (ITK style, i.e.: [x, y, z])
    pub fn set_spacing(&mut self, spacing: [f32; 3]) {
        assert!(spacing.iter().all(|&x| x > 0.0), "Spacing must > 0.");

        let mut affine = self.get_affine();

        let old_spacing = Array2::from_shape_vec(
            (1, 3),
            self.get_spacing().iter().map(|&x| x as f64).collect(),
        )
        .unwrap();
        let new_spacing =
            Array2::from_shape_vec((1, 3), spacing.iter().map(|&x| x as f64).collect()).unwrap();

        let rot_zoom = affine.slice(s![..3, ..3]);
        let result = &rot_zoom / &old_spacing * &new_spacing;
        affine.slice_mut(s![..3, ..3]).assign(&result);

        self.set_affine(affine);
    }

    /// Set Origin for Nifti1Image. (ITK style, i.e.: [x, y, z])
    pub fn set_origin(&mut self, origin: [f32; 3]) {
        let origin = [-origin[0], -origin[1], origin[2]];

        let mut affine = self.get_affine();

        let origin =
            Array2::from_shape_vec((3, 1), origin.iter().map(|&x| x as f64).collect()).unwrap();
        affine.slice_mut(s![..3, 3..4]).assign(&origin);

        self.set_affine(affine);
    }

    /// Set Direction for Nifti1Image. (ITK style, 3x3 list, i.e.: [[a,b,c], [d,e,f], [g,h,i]])
    pub fn set_direction(&mut self, direction: [[f32; 3]; 3]) {
        let direction = [
            -direction[0][0],
            -direction[0][1],
            -direction[0][2],
            -direction[1][0],
            -direction[1][1],
            -direction[1][2],
            direction[2][0],
            direction[2][1],
            direction[2][2],
        ]; // ITK style -> nifi-rs style

        let spacing = Array2::from_shape_vec(
            (1, 3),
            self.get_spacing().iter().map(|&x| x as f64).collect(),
        )
        .unwrap();

        let mut affine = self.get_affine();

        let direction =
            Array2::from_shape_vec((3, 3), direction.iter().map(|&x| x as f64).collect()).unwrap();

        let result = &direction * &spacing;
        affine.slice_mut(s![..3, ..3]).assign(&result);

        self.set_affine(affine);
    }

    /// Copy informations.
    pub fn copy_infomation(&mut self, im: &Nifti1Image<T>) {
        self.set_affine(im.get_affine());
    }

    // Pixel indices i,j,k -> Physical positions (ITK style, i.e.: [x, y, z])
    // No restriction on whether ijk or xyz are within the shape
    pub fn ijk2xyz(&self, ijk: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let [s_x, s_y, s_z] = self.get_spacing();
        let [o_x, o_y, o_z] = self.get_origin();
        ijk.par_iter()
            .map(|&[i, j, k]| [o_x + i * s_x, o_y + j * s_y, o_z + k * s_z])
            .collect()
    }

    // Physical positions -> Pixel indices i,j,k (ITK style, i.e.: [x, y, z])
    // No restriction on whether xyz or ijk are within the shape, no restriction on ijk being positive, please be careful
    pub fn xyz2ijk(&self, xyz: &[[f32; 3]]) -> Vec<[i32; 3]> {
        let [s_x, s_y, s_z] = self.get_spacing();
        let [o_x, o_y, o_z] = self.get_origin();
        xyz.par_iter()
            .map(|&[x, y, z]| {
                [
                    ((x - o_x) / s_x).round() as i32,
                    ((y - o_y) / s_y).round() as i32,
                    ((z - o_z) / s_z).round() as i32,
                ]
            })
            .collect()
    }

    /// Set default header for Nifti1Image. Equals:
    /// ```rust
    /// im.set_origin([0.0, 0.0, 0.0]);
    /// im.set_spacing([1.0, 1.0, 1.0]);
    /// im.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    pub fn set_default_header(&mut self) {
        self.set_origin([0.0, 0.0, 0.0]);
        self.set_spacing([1.0, 1.0, 1.0]);
        self.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
}

impl<T> fmt::Debug for Nifti1Image<T>
where
    T: DataElement + Pod,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Size: {:?}\n", self.get_size())?;
        write!(f, "Spacing: {:?}\n", self.get_spacing())?;
        write!(f, "Origin: {:?}\n", self.get_origin())?;
        write!(f, "Direction: {:?}\n", self.get_direction())
    }
}

/// Read image from disk.
pub fn read_image<T>(path: impl AsRef<Path>) -> Nifti1Image<T>
where
    T: DataElement + Pod,
{
    Nifti1Image::read(path)
}

/// Write image to disk.
pub fn write_image<T>(im: &Nifti1Image<T>, path: impl AsRef<Path>) -> ()
where
    T: DataElement + Pod,
{
    im.write(path);
}

/// Make a new Nifti1Image using array and affine like nibabel.
pub fn new<T>(ndarray: Array3<T>, affine: Array2<f64>) -> Nifti1Image<T>
where
    T: DataElement + Pod,
{
    let mut header = NiftiHeader::default();
    header.set_affine(&nd2na_4x4(affine.t().to_owned()));

    Nifti1Image { header, ndarray }
}

/// Get image from array with default header.
pub fn get_image_from_array<T>(ndarray: Array3<T>) -> Nifti1Image<T>
where
    T: DataElement + Pod,
{
    let affine: Array2<f64> = Array2::from_shape_vec(
        (4, 4),
        vec![
            -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    new(ndarray, affine)
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::error::Error;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_read_image() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let t = Instant::now();
        let img = read_image::<f32>(path);
        println!("Read Cost in Rust: {:?} ms", t.elapsed().as_millis());
        println!("Infos: {:?}", img);
        println!("Affine: {:?}", img.get_affine());
        Ok(())
    }

    #[test]
    fn test_write_image() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let img = read_image::<f32>(path);
        let t = Instant::now();
        write_image(
            &img,
            Path::new(r"test_data\results\test_write_image.nii.gz"),
        );
        println!("Write Cost in Rust: {:?} ms", t.elapsed().as_millis());
        Ok(())
    }

    #[test]
    fn test_read_attrs() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let img = read_image::<f32>(path);

        println!("size: {:?}", img.get_size());
        println!("spacing: {:?}", img.get_spacing());
        println!("origin: {:?}", img.get_origin());
        println!("direction: {:?}", img.get_direction());
        println!("affine: {:?}", img.get_affine());

        Ok(())
    }

    #[test]
    fn test_set_attrs() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let mut img = read_image::<f32>(path);

        println!("Before Image: {:?}", img);
        println!("Before Affine: {:?}", img.get_affine());
        println!("-----------------------------------------------");

        img.set_spacing([2, 3, 4].map(|x| x as f32));
        img.set_origin([23.5, -23.5, 117.5]);
        img.set_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        println!("After Image: {:?}", img);
        println!("After Affine: {:?}", img.get_affine());
        println!("-----------------------------------------------");

        write_image(&img, Path::new(r"test_data\results\test_set_attrs.nii.gz"));
        Ok(())
    }

    #[test]
    fn test_clone_and_copy_informations() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let img1 = read_image::<f32>(path);

        let mut img2 = img1.clone();
        img2.set_default_header();

        assert_eq!(img2.get_spacing(), [1.0, 1.0, 1.0]);
        assert_eq!(img2.get_origin(), [0.0, 0.0, 0.0]);
        assert_eq!(
            img2.get_direction(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );

        img2.copy_infomation(&img1);

        assert_eq!(img2.get_spacing(), img1.get_spacing());
        assert_eq!(img2.get_origin(), img1.get_origin());
        assert_eq!(img2.get_direction(), img1.get_direction());

        write_image(
            &img2,
            Path::new(r"test_data\results\test_clone_and_copy_informations.nii.gz"),
        );
        Ok(())
    }

    #[test]
    fn test_new() -> Result<(), Box<dyn Error>> {
        let path = Path::new(r"test_data\test.nii.gz");
        let img1 = read_image::<f32>(path);
        let affine1 = img1.get_affine();

        let vec = (0..27).map(|x| x as f32).collect();
        let arr = Array3::from_shape_vec((3, 3, 3), vec)?;

        let img2: Nifti1Image<f32> = new(arr, affine1);

        assert_eq!(img1.get_spacing(), img2.get_spacing());
        assert_eq!(img1.get_origin(), img2.get_origin());
        assert_eq!(img1.get_direction(), img2.get_direction());

        write_image(&img2, Path::new(r"test_data\results\test_new.nii.gz"));
        Ok(())
    }

    #[test]
    fn test_get_image_from_array() -> Result<(), Box<dyn Error>> {
        let vec = (0..24).map(|x| x as f32).collect();
        let arr = Array3::from_shape_vec((2, 3, 4), vec)?;

        let img: Nifti1Image<f32> = get_image_from_array(arr);
        write_image(
            &img,
            Path::new(r"test_data\results\test_get_image_from_array.nii.gz"),
        );
        Ok(())
    }
}
