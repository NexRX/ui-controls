use super::SearchResult;
use image::{flat::SampleLayout, DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::{s, Array2, ArrayView2, ShapeBuilder};
use rayon::iter::{ParallelBridge, ParallelIterator};

type LocatedNCC = (f32, (usize, usize));

// Helper function to convert image to ndarray array
// TODO: checkout docs for this: For conversion between ndarray, nalgebra and image check out nshare.
fn image_to_ndarray(image: ImageBuffer<Luma<f32>, Vec<f32>>) -> Array2<f32> {
    let SampleLayout {
        height,
        height_stride,
        width,
        width_stride,
        ..
    } = image.sample_layout();
    let shape = (width as usize, height as usize);
    let strides = (width_stride, height_stride);
    Array2::from_shape_vec(shape.strides(strides), image.into_raw()).unwrap()
}

fn compute_region_ncc(
    image: &ArrayView2<f32>,
    template: &ArrayView2<f32>,
    template_mean: f32,
    template_std: f32,
    (x, y): (usize, usize),
) -> LocatedNCC {
    let (tw, th) = template.dim();
    let region = image.slice(s![x..x + tw, y..y + th]);

    // Convert the region to f32 and compute the mean and standard deviation
    let region_mean = region.mean().unwrap();
    let region_std = region
        .mapv(|x| (x - region_mean).powi(2))
        .mean()
        .unwrap()
        .sqrt();

    // Compute the NCC score
    let numerator = ((region.mapv(|v| v - region_mean)) * (template - template_mean)).sum();
    let denominator = region_std * template_std * template.len() as f32;

    let ncc = if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    (ncc, (x, y))
}

fn compute_standard_deviation(image: &ArrayView2<f32>, mean: f32) -> f32 {
    (image - mean).mapv(|x| x.powi(2)).mean().unwrap().sqrt()
}

fn max_ncc(a: LocatedNCC, b: LocatedNCC) -> LocatedNCC {
    if a.0 > b.0 {
        return a;
    }
    b
}

fn normalize_cross_correlation(
    image: &ArrayView2<f32>,
    template: &ArrayView2<f32>,
) -> (f32, (usize, usize)) {
    let (iw, ih) = image.dim();
    let (tw, th) = template.dim();
    assert!(ih > th && iw > tw, "Template dimensions supercede images");

    let template_mean = template.mean().unwrap();
    let template_std = compute_standard_deviation(template, template_mean);

    let slide_range = s![0..(iw - tw), 0..(ih - th)];
    let image_slide_bounds = image.slice(slide_range);

    let best = image_slide_bounds
        .indexed_iter()
        .par_bridge()
        .map(|((x, y), _px)| {
            compute_region_ncc(image, &template, template_mean, template_std, (x, y))
        })
        .reduce(|| (0.0, (0, 0)), max_ncc);


    let best_percentage = (best.0 * 100.0).clamp(0.0, 100.0);

    (best_percentage, best.1)
}
pub fn find_target(
    image: &DynamicImage,
    template: &DynamicImage,
    confidence_threshold: f32,
) -> SearchResult {
    let result = normalize_cross_correlation(
        &image_to_ndarray(image.to_luma32f()).view(),
        &image_to_ndarray(template.to_luma32f()).view(),
    );

    Ok(Some((result.1 .0 as i32, result.1 .1 as i32)))
}

#[cfg(test)]
mod test_target {
    use super::*;
    use image::GrayImage;

    #[test]
    fn finds_button_at_high_confidence() {
        use approx::assert_relative_eq;

        let source = image::open("__fixtures__/screen-2k.png").unwrap();
        let target = image::open("__fixtures__/btn.png").unwrap();

        let (x, y) = super::find_target(&source, &target, 0.99999)
            .expect("An error occured")
            .expect("No cordinates found");

        assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
        assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
    }
}
