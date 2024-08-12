#[cfg(feature = "cpu")]
pub(crate) mod cpu;
#[cfg(feature = "imageproc")]
pub mod imageproc;
#[cfg(feature = "opencv")]
pub(crate) mod opencv;

use image::{DynamicImage, GenericImageView, Rgba};
use std::iter::zip;

type SearchResult = Result<Option<(i32, i32)>, SearchError>;

#[derive(Debug)]
pub enum SearchError {
    #[cfg(feature = "opencv")]
    OpenCV(::opencv::Error),
    Unexpected(String),
}

#[cfg(feature = "opencv")]
impl From<::opencv::Error> for SearchError {
    fn from(value: ::opencv::Error) -> Self {
        SearchError::OpenCV(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SearchAlgorithm {
    #[cfg(feature = "opencv")]
    OpenCV(f32),
    #[cfg(feature = "cpu")]
    CPU,
    #[cfg(feature = "imageproc")]
    ImageProc(f32),
    /// Brute force search algorithm with a specified **confidence** *(1.0 == strict match - 0.0 == anything)* for matching.
    BruteForce(f32),
}

impl SearchAlgorithm {
    pub fn new_recommended() -> Self {
        #[cfg(feature = "opencv")]
        return SearchAlgorithm::OpenCV(0.9);
        #[cfg(feature = "cpu")]
        return SearchAlgorithm::CPU;
        #[cfg(feature = "imageproc")]
        return SearchAlgorithm::ImageProc(0.9);
        #[cfg(not(any(feature = "opencv", feature = "imageproc")))]
        SearchAlgorithm::BruteForce(1.0)
    }

    pub fn find(&self, source: &mut DynamicImage, target: &DynamicImage) -> SearchResult {
        match self {
            SearchAlgorithm::BruteForce(confidence) => {
                Ok(brute_force::find_target(source, target, *confidence))
            }
            #[cfg(feature = "opencv")]
            SearchAlgorithm::OpenCV(confidence) => opencv::find_target(source, target, *confidence),
            #[cfg(feature = "cpu")]
            SearchAlgorithm::CPU => Ok(
                Some(cpu::template_matching(source, target).1).map(|v| (v.0 as i32, v.1 as i32))
            ),
            #[cfg(feature = "imageproc")]
            SearchAlgorithm::ImageProc(confidence) => {
                imageproc::find_target(source, target, *confidence)
            }
        }
    }
}

/// Finds a target/object on a image using brute force algorithm with a specified looseness.
pub(crate) mod brute_force {
    use image::{DynamicImage, GenericImageView};

    /// Finds the target image in the source image. Returning a tuple of the coordinates of the top left corner of the target image.
    /// - **source** - The source image to search for the target image.
    /// - **target** - The target image that we are looking for in the source image.
    /// - **confidence** - The percentage threshold of difference allowed between the two pixels based on 255. 0 being no strictness and 1 being perfect match.
    pub fn find_target(
        source: &mut DynamicImage,
        target: &DynamicImage,
        confidence: f32,
    ) -> Option<(i32, i32)> {
        let looseness = 1.0 - confidence;
        let (sw, sh) = source.dimensions();
        let (tw, th) = target.dimensions();

        let target_px = target.get_pixel(0, 0);

        for y in 0..sh - th {
            for x in 0..sw - tw {
                let current_px = source.get_pixel(x, y);

                if super::is_pixel_match(current_px, target_px, looseness) {
                    let crop = source.crop(x, y, tw, th);
                    if super::is_image_match(&crop, target, looseness) {
                        return Some((x as i32, y as i32));
                    }
                }
            }
        }

        None
    }

    #[cfg(test)]
    #[test]
    fn finds_button_strict() {
        let mut source = image::open("__fixtures__/screen-2k.png").unwrap();
        let target = image::open("__fixtures__/btn.png").unwrap();

        let is_found = find_target(&mut source, &target, 1.);
        assert_eq!(is_found, Some((1180, 934)));
    }

    #[cfg(test)]
    #[test]
    fn finds_nothing_for_random_image_strict() {
        let mut source = image::open("__fixtures__/screen-2k.png").unwrap();
        let target = image::open("__fixtures__/random-text.png").unwrap();

        let is_found = find_target(&mut source, &target, 1.);
        assert_eq!(is_found, None);
    }
}

/// Checks if two images are similar enough to be considered a match.
/// - **img1** - The first image to compare against the second one.
/// - **img2** - The second image to compare against the first one.
/// - **looseness** - The percentage of difference allowed between the two pixels based on 255. 0 being no difference and 1 being a perfect match
pub fn is_image_match(img1: &DynamicImage, img2: &DynamicImage, looseness: f32) -> bool {
    let (w1, h1) = img1.dimensions();

    for y in 0..h1 {
        for x in 0..w1 {
            let px1 = img1.get_pixel(x, y);
            let px2 = img2.get_pixel(x, y);
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
pub fn is_pixel_match(px1: Rgba<u8>, px2: Rgba<u8>, looseness: f32) -> bool {
    assert!((0.0..=1.0).contains(&looseness));
    let max_diff = (looseness * 255.0) as u8;

    for (c1, c2) in zip(px1.0, px2.0) {
        if c1.abs_diff(c2) > max_diff {
            return false;
        }
    }

    true
}
