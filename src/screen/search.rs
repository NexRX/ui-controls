use image::{DynamicImage, GenericImageView, Pixel, Rgb};
use std::iter::zip;

#[derive(Debug, Clone, Copy)]
pub enum SearchAlgorithm {
    /// Brute force search algorithm with a specified **looseness** *(0.0 == strict match - 1.0 == anything)* for matching.
    BruteForce(f32),
}

impl SearchAlgorithm {
    pub fn new_brute_force(looseness: f32) -> Self {
        SearchAlgorithm::BruteForce(looseness)
    }

    pub fn find(&self, source: &mut DynamicImage, target: &DynamicImage) -> Option<(u32, u32)> {
        match self {
            SearchAlgorithm::BruteForce(looseness) => find_target_bruteforce(source, target, *looseness),
        }
    }
}


/// Finds the target image in the source image. Returning a tuple of the coordinates of the top left corner of the target image.
/// - **source** - The source image to search for the target image.
/// - **target** - The target image that we are looking for in the source image.
/// - **looseness** - The percentage of difference allowed between the two pixels based on 255. 0 being no difference and 1 being a perfect match
pub fn find_target_bruteforce(
    source: &mut DynamicImage,
    target: &DynamicImage,
    looseness: f32,
) -> Option<(u32, u32)> {
    let (sw, sh) = source.dimensions();
    let (tw, th) = target.dimensions();

    let target_px = target.get_pixel(0, 0).to_rgb();

    for y in 0..sh - th {
        for x in 0..sw - tw {
            let current_px = source.get_pixel(x, y).to_rgb();

            if is_pixel_match(current_px, target_px, looseness) {
                let crop = source.crop(x, y, tw, th);
                if is_image_match(&crop, target, looseness) {
                    return Some((x, y));
                }
            }
        }
    }

    None
}

/// Finds a button on a screen using fixtures. Button is from https://codepen.io/lfdumas/pen/vOqarr.
#[cfg(test)]
mod brute_force {
    use super::find_target_bruteforce;

    #[test]
    fn finds_button_at_0_looseness() {
        let mut source = image::open("__fixtures__/screen-2k.png").unwrap();
        let target = image::open("__fixtures__/btn.png").unwrap();

        let is_found = find_target_bruteforce(&mut source, &target, 0.0);
        assert_eq!(is_found, Some((1180, 934)));
    }

    #[test]
    fn finds_nothing_for_random_image_at_0_looseness() {
        let mut source  = image::open("__fixtures__/screen-2k.png").unwrap();
        let target  = image::open("__fixtures__/random-text.png").unwrap();

        let is_found = find_target_bruteforce(&mut source, &target, 0.0);
        assert_eq!(is_found, None);
    }
}

/// Checks if two images are similar enough to be considered a match.
/// - **img1** - The first image to compare against the second one.
/// - **img2** - The second image to compare against the first one.
/// - **looseness** - The percentage of difference allowed between the two pixels based on 255. 0 being no difference and 1 being a perfect match
fn is_image_match(img1: &DynamicImage, img2: &DynamicImage, looseness: f32) -> bool {
    let (w1, h1) = img1.dimensions();

    for y in 0..h1 {
        for x in 0..w1 {
            let px1 = img1.get_pixel(x, y).to_rgb();
            let px2 = img2.get_pixel(x, y).to_rgb();
            if !is_pixel_match(px1, px2, looseness) {
                return false;
            }
        }
    }

    true
}

/// Checks if two pixels are similar enough to be considered a match
/// - **px1** - The first pixel to compare
/// - **px2** - The second pixel to compare
/// - **looseness** - The percentage of difference allowed between the two pixels based on 255. 0 being no difference and 1 being a perfect match
fn is_pixel_match(px1: Rgb<u8>, px2: Rgb<u8>, looseness: f32) -> bool {
    assert!((0.0..=1.0).contains(&looseness));
    let max_diff = (looseness * 255.0) as u8;

    for (c1, c2) in zip(px1.0, px2.0) {
        if c1.abs_diff(c2) > max_diff {
            return false;
        }
    }

    true
}
