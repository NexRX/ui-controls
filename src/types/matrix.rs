use derive_more as d;
use ndrustfft::Zero;
use rayon::iter::{
    Enumerate, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::{
    ops::*,
    slice::{Iter, IterMut},
};

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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.rows(), self.cols())
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
            cols: self.rows
        }
        
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
