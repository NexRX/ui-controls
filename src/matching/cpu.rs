use image::DynamicImage;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::vec;

use crate::types::matrix::Matrix;

// TODO: Clean up
// TODO: Implement alternative to NCC using Sum of Squared Differences (SSD)
// TODO: Look into the following for a potential speedup:
//In most cases, even prime-sized FFTs will be fast enough for your application. In the example of 5183 above, even that “slow” FFT only takes a few tens of microseconds to compute.
// https://docs.rs/rustfft/latest/rustfft/
// Some applications of the FFT allow for choosing an arbitrary FFT size (In many applications the size is pre-determined by whatever you’re computing).
// If your application supports choosing your own size, our advice is still to start by trying the size that’s most convenient to your application.
// If that’s too slow, see if you can find a nearby size whose prime factors are all 11 or smaller,
//                    \/-----------\/
//and you can expect a 2x-5x speedup. If that’s still too slow, find a nearby size whose prime factors are all 2 or 3, and you can expect a 1.1x-1.5x speedup.

fn fft2d(mut image: Matrix<Complex<f64>>) -> Matrix<Complex<f64>> {
    let (height, width) = image.dims();

    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(width);
    let fft_col = planner.plan_fft_forward(height);

    // Transpose the image for column FFTs (rows/height and cols/width are flipped for easier and better processing)
    let mut transposed_image: Matrix<Complex<f64>> = Matrix::new_zeros(width, height);

    // Transpose the image so that columns become rows
    transposed_image
        .par_enumerate_rows_mut()
        .for_each(|(col, transposed_row)| {
            for (i, val) in transposed_row.iter_mut().enumerate() {
                *val = image[(i, col)];
            }
        });

    // Apply FFT to each row in parallel
    let fn_row = move || {
        image.par_iter_rows_mut().for_each(|row| {
            fft_row.process(row);
        });
        image
    };

    let fn_col = || {
        // Apply FFT to each transposed row (original column) in parallel
        transposed_image.par_iter_rows_mut().for_each(|row| {
            fft_col.process(row);
        });

        transposed_image
    };

    let (mut image, transposed_image) = rayon::join(fn_row, fn_col);

    // Transpose back to the original layout
    image.par_enumerate_rows_mut().for_each(|(i, row)| {
        for (col, val) in row.iter_mut().enumerate() {
            *val = transposed_image[(col, i)];
        }
    });

    image
}

/// Computes the inverse 2D FFT of a 2D Vec.
fn ifft2d(mut frequency_domain: Matrix<Complex<f64>>) -> Matrix<Complex<f64>> {
    let (height, width) = frequency_domain.dims();

    let mut planner = FftPlanner::new();
    let ifft_row = planner.plan_fft_inverse(width);
    let ifft_col = planner.plan_fft_inverse(height);

    // Transpose the result for column FFTs
    let mut transposed_result: Vec<Vec<Complex<f64>>> = vec![vec![Complex::zero(); height]; width];

    transposed_result
        .par_iter_mut()
        .enumerate()
        .for_each(|(col, transposed_row)| {
            for (i, val) in transposed_row.iter_mut().enumerate() {
                *val = frequency_domain[(i, col)];
            }
        });

    // Apply inverse FFT to each row in parallel
    let fn_row = || {
        frequency_domain.par_iter_rows_mut().for_each(|row| {
            ifft_row.process(row);
        });
        frequency_domain
    };

    // Apply inverse FFT to each transposed row (original column) in parallel
    let fn_col = || {
        transposed_result.par_iter_mut().for_each(|row| {
            ifft_col.process(row);
        });
        transposed_result
    };

    let (mut frequency_domain, transposed_result) = rayon::join(fn_row, fn_col);

    // Transpose back to the original layout
    frequency_domain
        .par_enumerate_rows_mut()
        .for_each(|(i, row)| {
            for (col, val) in row.iter_mut().enumerate() {
                *val = transposed_result[col][i];
            }
        });

    // Normalize the result
    let normalization_factor = (width * height) as f64;
    frequency_domain.par_iter_rows_mut().for_each(|row| {
        row.iter_mut().for_each(|val| {
            *val /= normalization_factor;
        });
    });

    frequency_domain
}

/// Convert an image to a 2D vector of Complex numbers (grayscale values).
fn image_to_complex_matrix(image: &DynamicImage) -> Matrix<Complex<f64>> {
    let image = image.to_luma8();
    let (width, height) = image.dimensions();

    let mut result = Matrix::<Complex<f64>>::new_zeros(height as usize, width as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).0[0] as f64;
            result[(y as usize, x as usize)] = Complex::new(pixel, 0.0);
        }
    }

    result
}

fn compute_cross_spectrum(
    ga: &Matrix<Complex<f64>>,
    gb: &Matrix<Complex<f64>>,
) -> Matrix<Complex<f64>> {
    let (height, width) = ga.dims();
    let mut result = Matrix::new_zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let conj_gb = gb[(y, x)].conj();
            let numerator = ga[(y, x)] * conj_gb;
            let denominator = numerator.norm();
            result[(y, x)] = if denominator != 0.0 {
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
fn find_peak_correlation(
    correlation: &Matrix<Complex<f64>>,
    normalized: bool,
) -> (f64, (usize, usize)) {
    let (height, width) = correlation.dims();

    let peak = (0..height)
        .map(|y| {
            (0..width)
                .map(move |x| (correlation[(y, x)].re, (x, y)))
                .reduce(max_fft)
        })
        .map(|v| v.unwrap_or_default())
        .reduce(max_fft)
        .unwrap_or_default();

    if !normalized {
        return peak;
    }

    let max = correlation
        .par_iter_rows()
        .map(|row| {
            row.par_iter()
                .map(|v| v.re)
                .reduce(|| 0f64, |a, b| a.max(b))
        })
        .reduce(|| 0f64, |a, b| a.max(b));

    let min = correlation
        .par_iter_rows()
        .map(|row| {
            row.par_iter()
                .map(|v| v.re)
                .reduce(|| 0f64, |a, b| a.max(b))
        })
        .reduce(|| 0f64, |a, b| a.min(b));

    let peak_norm = (peak.0 - min) / (max - min);
    (peak_norm, peak.1)
}

fn pad_image(image: &Matrix<Complex<f64>>, width: usize, height: usize) -> Matrix<Complex<f64>> {
    let mut padded = Matrix::new_zeros(height, width);
    let (og_height, og_width) = image.dims();

    for y in 0..og_height {
        for x in 0..og_width {
            padded[(y, x)] = image[(y, x)];
        }
    }

    padded
}

pub fn template_matching(source: &DynamicImage, template: &DynamicImage) -> (f64, (usize, usize)) {
    let (w, h) = (source.width() as usize, source.height() as usize);

    // Pad the template image to match the background dimensions
    let source = image_to_complex_matrix(source);
    let template = pad_image(&image_to_complex_matrix(template), w, h);

    let fn_ga = || fft2d(source);
    let fn_gb = || fft2d(template);

    let (ga, gb) = rayon::join(fn_ga, fn_gb);

    let cc = compute_cross_spectrum(&ga, &gb);

    let correlation = ifft2d(cc);
    find_peak_correlation(&correlation, true)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn find_btn_on_screen() {
        let source = image::open("__fixtures__/screen-2k.png").unwrap();
        let template = image::open("__fixtures__/btn.png").unwrap();

        let time = std::time::Instant::now();
        let (score, (x, y)) = template_matching(&source, &template);
        println!("{:?}", time.elapsed());
        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(score, 1., max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }

    #[test]
    fn find_btn_on_screen_pow2() {
        let source = image::open("__fixtures__/screen-pow2.png").unwrap();
        let template = image::open("__fixtures__/gb-pow2-btn.png").unwrap();

        let time = std::time::Instant::now();
        let (score, (x, y)) = template_matching(&source, &template);
        println!("{:?}", time.elapsed());
        assert_relative_eq!(x as f32, 1608f32, max_relative = 0.0017);
        assert_relative_eq!(score, 1., max_relative = 0.0017);
        assert_relative_eq!(y as f32, 714f32, max_relative = 0.0017);
    }
}
