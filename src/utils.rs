use nalgebra::{Matrix4, Scalar};
use ndarray::prelude::*;

pub fn nd2na_4x4<T>(arr: ndarray::Array2<T>) -> Matrix4<T>
where
    T: Clone + Scalar,
{
    let vec: Vec<T> = arr.iter().cloned().collect();
    Matrix4::from_vec(vec)
}

pub fn na2nd_4x4<T>(arr: Matrix4<T>) -> ndarray::Array2<T>
where
    T: Clone + Scalar,
{
    Array2::from_shape_vec((arr.nrows(), arr.ncols()), arr.as_slice().to_vec()).unwrap()
}
