use derive_more as d;
use image::{DynamicImage, GenericImageView, ImageResult};
use ndrustfft::{Complex, Zero};
use rayon::iter::{
    Enumerate, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::{
    ops::*,
    slice::{Iter, IterMut},
};

pub type MatrixImage = Matrix<f64>;
pub type MatrixImage32 = Matrix<f32>;
pub type MatrixImageC = Matrix<Complex<f64>>;

#[derive(d::AsRef, d::AsMut, Debug, Clone, PartialEq, d::From, d::Into)]
#[from(forward)]
pub struct Matrix<T> {
    #[as_ref]
    #[as_mut]
    #[into]
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    // Create a new Matrix
    pub fn new(rows: usize, cols: usize, init: T) -> Matrix<T>
    where
        T: Copy,
    {
        Matrix {
            data: vec![init; rows * cols],
            rows,
            cols,
        }
    }

    pub fn new_zeros(rows: usize, cols: usize) -> Matrix<T>
    where
        T: Zero + Copy,
    {
        Matrix::new(rows, cols, T::zero())
    }

    // Uses the default function of T to create a (symboliclly) empty Matrix
    pub fn new_default(rows: usize, cols: usize) -> Matrix<T>
    where
        T: Default + Copy,
    {
        Matrix {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    pub fn new_raw(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {
        Matrix { data, rows, cols }
    }

    pub fn into_raw(self) -> Vec<T> {
        self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the dimensions of this matrix as a tuple of (rows, cols) or in the context of images (height, width).
    pub fn dims(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Panics on casting usize to u32, best use dims unless your casting anyway
    pub fn dims_u32(&self) -> (u32, u32) {
        (self.rows() as u32, self.cols() as u32)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Get the element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    // Get a mutable reference to the element at (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Used for GPU buffer purposes. Contents of a buffer on creation. Executes `bytemuck::cast_slice`
    pub fn cast_slice(&self) -> &[u8]
    where
        T: bytemuck::Pod,
    {
        bytemuck::cast_slice(&self.data)
    }

    pub fn enumerate_element(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        (0..self.rows()).flat_map(move |row| {
            (0..self.cols()).map(move |col| ((row, col), &self.data[row * self.cols + col]))
        })
    }

    // Iterate over rows
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols)
    }

    // Iterate over columns
    pub fn iter_cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.cols)
            .map(move |col| (0..self.rows).map(move |row| &self.data[row * self.cols + col]))
    }

    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_mut(self.cols)
    }

    // fn iter_cols_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
    //     self.data.iter().step_by(step)
    // }

    pub fn into_transpose(self) -> Self
    where
        T: Copy + Zero,
    {
        let mut data = vec![T::zero(); self.rows * self.cols];
        transpose::transpose(&self.data, &mut data, self.cols, self.rows);
        Self {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn into_complex(self) -> Matrix<Complex<T>>
    where
        T: Copy + Zero,
    {
        let (rows, cols) = self.dims();
        let data = self
            .data
            .into_iter()
            .map(|v| Complex::new(v, T::zero()))
            .collect::<Vec<Complex<T>>>();
        Matrix::<Complex<T>> { data, rows, cols }
    }

    #[cfg(test)]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, file_path: P)
    where
        T: std::fmt::Display,
    {
        use std::fs;
        use std::io::Write;

        // Extract the parent directory from the file path
        if let Some(parent_dir) = file_path.as_ref().parent() {
            // Create the parent directory and any necessary subdirectories
            fs::create_dir_all(parent_dir).expect("Failed to create parent dirs");
        }

        // Create the file
        let mut file = fs::File::create(file_path).expect("Failed to open file");

        for row in self.iter_rows() {
            let line = row
                .iter()
                .map(|v| v.to_string())
                .reduce(|a, b| format!("{a} {b}"))
                .expect("Missing elements");
            writeln!(&mut file, "{line}").expect("Failed to write to file");
        }
    }
}

impl From<DynamicImage> for Matrix<f32> {
    fn from(value: DynamicImage) -> Self {
        let (cols, rows) = value.dimensions();
        let size = value.to_luma32f().len();
        let data: Vec<f32> = value.to_luma32f().to_vec();
        assert_eq!(size, data.len());
        assert_eq!(data.len(), (cols * rows) as usize);
        Self::new_raw(rows as usize, cols as usize, data)
    }
}

impl From<&DynamicImage> for Matrix<f32> {
    fn from(value: &DynamicImage) -> Self {
        let (cols, rows) = value.dimensions();
        let data: Vec<f32> = value.to_luma_alpha32f().to_vec();
        Self::new_raw(rows as usize, cols as usize, data)
    }
}

impl From<DynamicImage> for Matrix<f64> {
    fn from(value: DynamicImage) -> Self {
        Matrix::<f32>::from(value).into_f64()
    }
}

impl From<&DynamicImage> for Matrix<f64> {
    fn from(value: &DynamicImage) -> Self {
        Matrix::<f32>::from(value).into_f64()
    }
}

impl Matrix<f32> {
    pub fn open_image(image_path: &str) -> ImageResult<Self> {
        let image = image::open(image_path)?;
        Ok(Self::from(image))
    }

    pub fn into_f64(self) -> Matrix<f64> {
        let (rows, cols) = self.dims();
        let data: Vec<f64> = self.into_raw().into_iter().map(|v| v as f64).collect();
        Matrix::<f64>::new_raw(rows, cols, data)
    }
}

impl Matrix<f64> {
    pub fn open_image(image_path: &str) -> ImageResult<Self> {
        Matrix::<f32>::open_image(image_path).map(|v| v.into_f64())
    }
}

impl<T: Send + Sync> Matrix<T> {
    pub fn par_iter_rows(&self) -> impl ParallelIterator<Item = &[T]> {
        self.data.as_parallel_slice().par_chunks(self.rows)
    }

    pub fn par_iter_cols(&self) -> impl ParallelIterator<Item = impl ParallelIterator<Item = &T>> {
        (0..self.cols).into_par_iter().map(move |col| {
            (0..self.rows)
                .into_par_iter()
                .map(move |row| &self.data[row * self.cols + col])
        })
    }

    pub fn par_iter_rows_mut(&mut self) -> impl ParallelIterator<Item = &mut [T]> {
        self.data.par_chunks_mut(self.cols)
    }

    pub fn par_enumerate_rows_mut(&mut self) -> Enumerate<rayon::slice::ChunksMut<'_, T>> {
        self.data.par_chunks_mut(self.cols).enumerate()
    }

    // pub fn par_iter_cols_mut(
    //     &mut self,
    // ) -> impl ParallelIterator<Item = impl ParallelIterator<Item = &mut T>> {
    //     (0..self.cols).into_par_iter().map(move |col| {
    //         // Collect mutable references into a local vector to avoid lifetime issues
    //         (0..self.rows)
    //             .into_par_iter()
    //             .map(move |row| &mut self.data[row * self.cols + col])
    //     })
    // }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(v: Vec<Vec<T>>) -> Matrix<T> {
        let rows = v.len();
        let cols = v.first().map_or(0, |r| r.len());

        // Flatten the 2D vector into a 1D vector
        let data: Vec<T> = v.into_iter().flatten().collect();

        Matrix { data, rows, cols }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[row * self.cols + col]
    }
}

impl<T> Index<(u32, u32)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        Index::<(usize, usize)>::index(self, (index.0 as usize, index.1 as usize))
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(u32, u32)> for Matrix<T> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        IndexMut::<(usize, usize)>::index_mut(self, (index.0 as usize, index.1 as usize))
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T> AsRef<[T]> for Matrix<T> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T> AsMut<[T]> for Matrix<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// Rayon Support
impl<T> IntoParallelIterator for Matrix<T>
where
    T: Send,
{
    type Item = T;
    type Iter = rayon::vec::IntoIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        self.data.into_par_iter()
    }
}

impl<'a, T> IntoParallelRefIterator<'a> for Matrix<T>
where
    T: Sync + 'a,
{
    type Item = &'a T;
    type Iter = rayon::slice::Iter<'a, T>;

    fn par_iter(&'a self) -> Self::Iter {
        self.data.par_iter()
    }
}

impl<'a, T> IntoParallelRefMutIterator<'a> for Matrix<T>
where
    T: Send + 'a,
{
    type Item = &'a mut T;
    type Iter = rayon::slice::IterMut<'a, T>;

    fn par_iter_mut(&'a mut self) -> Self::Iter {
        self.data.par_iter_mut()
    }
}

#[allow(clippy::extra_unused_lifetimes)]
impl<'__, T> ParallelSlice<T> for Matrix<T>
where
    T: Sync,
{
    fn as_parallel_slice(&self) -> &[T] {
        self.data.as_parallel_slice()
    }
}

#[allow(clippy::extra_unused_lifetimes)]
impl<'__, T> ParallelSliceMut<T> for Matrix<T>
where
    T: Send,
{
    fn as_parallel_slice_mut(&mut self) -> &mut [T] {
        self.data.as_parallel_slice_mut()
    }
}
