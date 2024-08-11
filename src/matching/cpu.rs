use image::{DynamicImage, GenericImageView};
use ndarray::parallel::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::vec;

type RawImage<T = f32> = Vec<Vec<T>>;

trait ImageVec<T> {
    fn into_complex(self) -> RawImage<Complex<T>>;

    fn dimensions(&self) -> (usize, usize);
}

// Trait to find the maximum value
trait ToReal<T> {
    fn to_real(&self) -> T;
}

// Implement MaxValue for f64
impl ToReal<f64> for f64 {
    fn to_real(&self) -> f64 {
        *self
    }
}

// Implement MaxValue for Complex<f64>
impl ToReal<f64> for Complex<f64> {
    fn to_real(&self) -> f64 {
        self.re
    }
}

impl<T: Copy> ImageVec<T> for Vec<Vec<T>> {
    fn into_complex(self) -> RawImage<Complex<T>> {
        self.into_iter()
            .map(|v| v.into_iter().map(|v| Complex { re: v, im: v }).collect())
            .collect()
    }

    fn dimensions(&self) -> (usize, usize) {
        let rows = self.len();
        let cols = self[0].len();
        (rows, cols)
    }
}
fn fft2d(mut image: RawImage<Complex<f64>>) -> RawImage<Complex<f64>> {
    let height = image.len();
    let width = image[0].len();

    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(width);
    let fft_col = planner.plan_fft_forward(height);

    // Apply FFT to each row in parallel
    image.par_iter_mut().for_each(|row| {
        fft_row.process(row);
    });

    // Transpose the image for column FFTs
    let mut transposed_image: RawImage<Complex<f64>> =
        vec![vec![Complex::default(); height]; width];

    // Transpose the image so that columns become rows
    transposed_image
        .par_iter_mut()
        .enumerate()
        .for_each(|(col, transposed_row)| {
            for (i, val) in transposed_row.iter_mut().enumerate() {
                *val = image[i][col];
            }
        });

    // Apply FFT to each transposed row (original column) in parallel
    transposed_image.par_iter_mut().for_each(|row| {
        fft_col.process(row);
    });

    // Transpose back to the original layout
    image.par_iter_mut().enumerate().for_each(|(i, row)| {
        for (col, val) in row.iter_mut().enumerate() {
            *val = transposed_image[col][i];
        }
    });

    image
}

/// Computes the inverse 2D FFT of a 2D Vec.
fn ifft2d(mut frequency_domain: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let height = frequency_domain.len();
    let width = frequency_domain[0].len();

    let mut planner = FftPlanner::new();
    let ifft_row = planner.plan_fft_inverse(width);
    let ifft_col = planner.plan_fft_inverse(height);

    // Apply inverse FFT to each row in parallel
    frequency_domain.par_iter_mut().for_each(|row| {
        ifft_row.process(row);
    });

    // Transpose the result for column FFTs
    let mut transposed_result: Vec<Vec<Complex<f64>>> = vec![vec![Complex::zero(); height]; width];

    transposed_result
        .par_iter_mut()
        .enumerate()
        .for_each(|(col, transposed_row)| {
            for (i, val) in transposed_row.iter_mut().enumerate() {
                *val = frequency_domain[i][col];
            }
        });

    // Apply inverse FFT to each transposed row (original column) in parallel
    transposed_result.par_iter_mut().for_each(|row| {
        ifft_col.process(row);
    });

    // Transpose back to the original layout
    frequency_domain
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            for (col, val) in row.iter_mut().enumerate() {
                *val = transposed_result[col][i];
            }
        });

    // Normalize the result
    let normalization_factor = (width * height) as f64;
    frequency_domain.par_iter_mut().for_each(|row| {
        row.iter_mut().for_each(|val| {
            *val /= normalization_factor;
        });
    });

    frequency_domain
}

/// Convert an image to a 2D vector of Complex numbers (grayscale values).
fn image_to_complex_vec(image: &DynamicImage) -> RawImage<Complex<f64>> {
    let image = image.to_luma8();
    let (width, height) = image.dimensions();

    let mut result = vec![vec![Complex::new(0.0, 0.0); width as usize]; height as usize];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).0[0] as f64;
            result[y as usize][x as usize] = Complex::new(pixel, 0.0);
        }
    }

    result
}

fn compute_cross_spectrum(
    ga: &RawImage<Complex<f64>>,
    gb: &RawImage<Complex<f64>>,
) -> RawImage<Complex<f64>> {
    let height = ga.len();
    let width = ga[0].len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); width]; height];

    for y in 0..height {
        for x in 0..width {
            let conj_gb = gb[y][x].conj();
            let numerator = ga[y][x] * conj_gb;
            let denominator = numerator.norm();
            result[y][x] = if denominator != 0.0 {
                numerator / denominator
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    result
}

fn max_fft(a: (f64, (usize, usize)), b: (f64, (usize, usize))) -> (f64, (usize, usize)) {
    if a.0 > b.0 {
        a
    } else {
        b
    }
}

// Function to find the peak correlation
fn find_peak_correlation<T>(correlation: &RawImage<T>) -> (f64, (usize, usize))
where
    T: ToReal<f64> + Copy,
{
    let height = correlation.len();
    let width = correlation[0].len();

    (0..height)
        .map(|y| {
            (0..width)
                .map(move |x| (correlation[y][x].to_real(), (x, y)))
                .reduce(max_fft)
        })
        .map(|v| v.unwrap_or_default())
        .reduce(max_fft)
        .unwrap_or_default()
}

fn pad_image(
    image: &RawImage<Complex<f64>>,
    width: usize,
    height: usize,
) -> RawImage<Complex<f64>> {
    let mut padded = vec![vec![Complex::new(0.0, 0.0); width]; height];

    let original_height = image.len();
    let original_width = image[0].len();

    for y in 0..original_height {
        for x in 0..original_width {
            padded[y][x] = image[y][x];
        }
    }

    padded
}

fn template_matching(source: &DynamicImage, template: &DynamicImage) -> (f64, (usize, usize)) {
    let (w, h) = (source.width() as usize, source.height() as usize);

    // Pad the template image to match the background dimensions
    let source = image_to_complex_vec(source);
    let template = pad_image(&image_to_complex_vec(template), w, h);

    let ga = fft2d(source);
    let gb = fft2d(template);

    let cross_spectrum = compute_cross_spectrum(&ga, &gb);

    let correlation = ifft2d(cross_spectrum);
    find_peak_correlation(&correlation)
}

fn find_target(
    source: &DynamicImage,
    template: &DynamicImage,
    confidence_threshold: f64,
) -> Option<(usize, usize)> {
    let (score, xy) = template_matching(source, template);

    if score < confidence_threshold {
        return None;
    }

    Some(xy)
}

fn normalize_correlation(
    correlation: &RawImage<Complex<f64>>,
    min: f64,
    max: f64,
) -> RawImage<f64> {
    let height = correlation.len();
    let width = correlation[0].len();
    let mut result = vec![vec![0.0; width]; height];

    for y in 0..height {
        for x in 0..width {
            let value = correlation[y][x].re;
            result[y][x] = if max != min {
                (value - min) / (max - min)
            } else {
                0.0 // Handle the case where all values are the same
            };
        }
    }

    result
}

fn find_min_max(correlation: &RawImage<Complex<f64>>) -> (f64, f64) {
    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for row in correlation.iter() {
        for &value in row.iter() {
            let re = value.re;
            if re < min {
                min = re;
            }
            if re > max {
                max = re;
            }
        }
    }

    (min, max)
}

fn calculate_percentage_score(peak_value: f64) -> f64 {
    let max_possible_value = 1.0; // Assuming the max correlation value is normalized to 1.0
    (peak_value / max_possible_value)
}

// TODO: Add a normalize option so we can give a percentage threshold to determine if the correlation is good enough.

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn find_btn_on_screen() {
        let source = image::open("__fixtures__/screen-2k.png").unwrap();
        let template = image::open("__fixtures__/btn.png").unwrap();

        let (score, (x, y)) = template_matching(&source, &template);
        assert_relative_eq!(score, 1., max_relative = 0.0017);
        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }
}
