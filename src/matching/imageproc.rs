use image::DynamicImage;
use imageproc::template_matching::{
    find_extremes, match_template_parallel, Extremes, MatchTemplateMethod,
};

use super::SearchResult;

pub fn find_target(
    haystack_img: &DynamicImage,
    needle_img: &DynamicImage,
    confidence_threshold: f32,
) -> SearchResult {
    let result = match_template_parallel(
        &haystack_img.to_luma8(),
        &needle_img.to_luma8(),
        MatchTemplateMethod::CrossCorrelationNormalized,
    );

    let Extremes {
        max_value,
        max_value_location,
        ..
    } = find_extremes(&result);

    if max_value < confidence_threshold {
        return Ok(None);
    }

    Ok(Some((
        max_value_location.0 as i32,
        max_value_location.1 as i32,
    )))
}

#[cfg(test)]
#[test]
fn finds_button_at_high_confidence() {
    use approx::assert_relative_eq;

    let source = image::open("__fixtures__/screen-2k.png").unwrap();
    let target = image::open("__fixtures__/btn.png").unwrap();

    let (x, y) = find_target(&source, &target, 0.99999)
        .expect("An error occured")
        .expect("No cordinates found");

    assert_relative_eq!(x as f32, 1180f32, max_relative = 0.0017);
    assert_relative_eq!(y as f32, 934f32, max_relative = 0.0017);
}
