use super::SearchResult;
use image::{flat::SampleLayout, DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::{s, Array2, ArrayView2, ShapeBuilder};
use ndrustfft::*;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::path::Path;

// TODO: Comething seems wrong with how we are iterating, things stick to the left especiallly with template. 
// Check it out, maybe we need to do something about it, we willl see

fn load_image_to_complex_array(file_path: &str) -> Array2<Complex<f32>> {
    let img = image::open(file_path).expect("Failed to open image");
    let gray_img = img.to_luma32f();
    let (width, height) = gray_img.dimensions();
    let mut complex_array = Array2::<Complex<f32>>::zeros((width as usize, height as usize));

    for (x, y, luma) in gray_img.enumerate_pixels() {
        let pixel_value = luma[0];
        complex_array[(x as usize, y as usize)] = Complex::new(pixel_value, 0.0);
    }

    complex_array
}

// Pad array to a specific size
fn pad_to_size(array: &Array2<Complex<f32>>, new_size: (usize, usize)) -> Array2<Complex<f32>> {
    let (w, h) = array.dim();
    let (new_w, new_h) = new_size;

    let mut padded = Array2::<Complex<f32>>::zeros((new_w, new_h));

    for x in 0..w {
        for y in 0..h {
            padded[(x, y)] = array[(x, y)];
        }
    }

    padded
}

// Perform 2D FFT
fn fft2d(data: &Array2<f32>) -> Array2<Complex<f32>> {
    let (w, h) = data.dim();
    let mut output = Array2::<Complex<f32>>::zeros(((w / 2) + 1, h));
    let planner = R2cFftHandler::<f32>::new(w);

    ndfft_r2c_par(data, &mut output, &planner, 0);

    output
}

// Perform 2D IFFT
fn ifft2d(data: &Array2<Complex<f32>>, original_dim: Option<(usize, usize)>) -> Array2<f32> {
    let (w, h) = data.dim();

    // Adjust the dimensions to match the original image size (before padding)
    let mut output = Array2::<f32>::zeros(original_dim.unwrap_or((w * 2 - 2, h * 2 - 2)));

    let planner = R2cFftHandler::<f32>::new(output.dim().0);
    // r2c actually means complex to real since its ifft
    ndifft_r2c_par(data, &mut output, &planner, 0);

    output
}

// Perform cross-correlation
fn cross_correlation(
    target: &Array2<Complex<f32>>,
    template: &Array2<Complex<f32>>,
) -> Array2<f32> {
    let (tw, th) = template.dim();
    let (w, h) = target.dim();

    // Pad template to match the size of the target
    let padded_template = pad_to_size(template, (w, h));

    // Compute FFT of target and template

    let target_fft = fft2d(&Array2::<f32>::from(target.mapv(|c| c.re)));
    save_arrayc(&target_fft, "target/target_ftt.png");
    let template_fft = fft2d(&Array2::<f32>::from(padded_template.mapv(|c| c.re)));
    save_arrayc(&template_fft, "target/template_fft.png");

    // Compute cross-correlation in frequency domain
    let template_fft_conj = template_fft.mapv(|c| c.conj());
    let mut result_fft = target_fft * template_fft_conj;
    save_arrayc(&result_fft, "target/result_fft.png");
    
    // Perform IFFT to obtain the cross-correlation result
    let result_ifft = ifft2d(&result_fft, Some(target.dim()));
    save_array(&result_ifft, "target/result_ifft.png");

    result_ifft
}

// Find the position of the template in the target image
fn find_template_position(correlation: &Array2<f32>) -> (usize, usize) {
    let (w, h) = correlation.dim();
    let mut max_val = f32::MIN;
    let mut max_pos = (0, 0);

    for ((x, y), &val) in correlation.indexed_iter() {
        if val > max_val {
            max_val = val;
            max_pos = (x, y);
        }
    }

    max_pos
}

fn compute_template_position(
    target: &Array2<Complex<f32>>,
    template: &Array2<Complex<f32>>,
) -> (usize, usize) {
    let correlation = cross_correlation(target, template);
    save_array(&correlation, "target/correlation.png");
    find_template_position(&correlation)
}

fn image_to_ndarray(image: ImageBuffer<Luma<f32>, Vec<f32>>) -> Array2<f32> {
    // Create an ndarray from the image data
    let ndarray = Array2::<f32>::from_shape_vec(
        (image.height() as usize, image.width() as usize),
        image.into_raw(),
    )
    .expect("Failed to create ndarray from image data");

    ndarray
}

fn open_image_array(path: &str) -> Array2<f32> {
    let image = image::open(path).expect("Failed to open image");
    image_to_ndarray(image.to_luma32f())
}

fn array_to_image(ndarray: &Array2<f32>) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (height, width) = ndarray.dim();

    // Create an ImageBuffer with the same dimensions as the ndarray
    let mut img = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width as u32, height as u32);

    // Fill the image buffer with the ndarray data
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(x as u32, y as u32, Luma::from([ndarray[[y, x]]]));
        }
    }

    img
}

fn save_array(ndarray: &Array2<f32>, path: &str) {
    let image = array_to_image(ndarray);
    let (width, height) = image.dimensions();
    
    // Create a new image buffer to hold the u8 data
    let mut image_u8 = ImageBuffer::new(width, height);

    // Iterate over each pixel
    for (x, y, pixel_f32) in image.enumerate_pixels() {
        // Convert the f32 value to u8
        let Luma([value_f32]) = *pixel_f32;
        let value_u8 = (value_f32 * 255.0).clamp(0.0, 255.0) as u8;
        
        // Set the pixel value in the new image buffer
        image_u8.put_pixel(x, y, Luma([value_u8]));
    }
    image_u8.save(path).expect("Failed to save image");
}

fn save_arrayc(ndarray: &Array2<Complex<f32>>, path: &str) {
   save_array(&ndarray.map(|v| v.re), path);
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn finds_button_at_high_confidence() {
        let source = open_image_array("__fixtures__/screen-2k.png").map(|&x| Complex::new(x, 0.0));
        save_array(&source.mapv(|v| v.re), "target/source.png");
        let target = open_image_array("__fixtures__/btn.png").map(|&x| Complex::new(x, 0.0));
        save_array(&target.mapv(|v| v.re), "target/target.png");

        let (x,y) = compute_template_position(&source, &target);

        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }
    // #[test]
    // fn finds_button_at_high_confidence() {
    //     use approx::assert_relative_eq;

    //     let source = image::open("__fixtures__/screen-2k.png").unwrap();
    //     let target = image::open("__fixtures__/btn.png").unwrap();

    //     let (x, y) = super::find_target(&source, &target, 0.99999)
    //         .expect("An error occured")
    //         .expect("No cordinates found");

    //     assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
    //     assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    // }
}
