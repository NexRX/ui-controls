use image::{DynamicImage, GenericImageView};
use opencv::core::{min_max_loc, no_array, Point, Scalar, CV_32FC1, CV_8UC3};
use opencv::imgproc::{self, match_template};
use opencv::prelude::*;

use super::SearchResult;

fn dynamic_image_to_mat(img: &DynamicImage) -> opencv::Result<Mat> {
    let (width, height) = img.dimensions();
    let img_rgb = img.to_rgb8();

    // Create a Mat directly from the RGB image data
    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height as i32,
            width as i32,
            CV_8UC3,
            img_rgb.as_ptr() as *mut std::ffi::c_void,
            opencv::core::Mat_AUTO_STEP,
        )
    }?;

    Ok(mat)
}

fn template_matching(source: &Mat, template: &Mat) -> opencv::Result<(f64, (i32, i32))> {
    // Create result matrix for template matching using new_rows_cols_with_default
    let result_cols = source.cols() - template.cols() + 1;
    let result_rows = source.rows() - template.rows() + 1;
    let mut result =
        Mat::new_rows_cols_with_default(result_rows, result_cols, CV_32FC1, Scalar::all(0.0))?;

    // Perform template matching
    match_template(
        &source,
        &template,
        &mut result,
        imgproc::TM_CCOEFF_NORMED,
        &no_array(),
    )?;

    // Find the global max value and location in result matrix
    // let mut min_val = 0.0;
    let mut max_val = 0.0;
    // let mut min_loc = Point::new(0, 0);
    let mut max_loc = Point::new(0, 0);
    min_max_loc(
        &result,
        None,
        Some(&mut max_val),
        None,
        Some(&mut max_loc),
        &no_array(),
    )?;

    Ok((max_val, (max_loc.x, max_loc.y)))
}

pub fn find_target(
    source: &DynamicImage,
    template: &DynamicImage,
    confidence_threshold: f32,
) -> SearchResult {
    // Convert images to OpenCV Mat
    let source = dynamic_image_to_mat(source)?;
    let template = dynamic_image_to_mat(template)?;

    let result = template_matching(&source, &template)?;

    if result.0 >= confidence_threshold as f64 {
        return Ok(Some(result.1));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_button_on_screen() {
        use approx::assert_relative_eq;

        let source = image::open("__fixtures__/screen-2k.png").unwrap();
        let template = image::open("__fixtures__/btn.png").unwrap();
        let source = dynamic_image_to_mat(&source).unwrap();
        let template = dynamic_image_to_mat(&template).unwrap();

        let time = std::time::Instant::now();
        let (_score, (x, y)) = template_matching(&source, &template).expect("An error occured");
        println!("{:?}", time.elapsed());
        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }

    #[test]
    fn finds_button_at_high_confidence() {
        use approx::assert_relative_eq;

        let source = image::open("__fixtures__/screen-2k.png").unwrap();
        let target = image::open("__fixtures__/btn.png").unwrap();

        let time = std::time::Instant::now();
        let (x, y) = find_target(&source, &target, 0.99999)
            .expect("An error occured")
            .expect("No cordinates found");
        println!("{:?}", time.elapsed());
        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }
}
