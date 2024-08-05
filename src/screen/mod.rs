mod search;

use derive_more::{Deref, DerefMut};
use image::DynamicImage;
use xcap::Monitor;

pub trait ScreenControls {
    /// Returns the width and height of the screen.
    fn size(&self) -> (u32, u32);

    /// Takes a image of the screen and returns the (dynamic) image in memory
    fn screenshot(&self) -> DynamicImage;
    // TODO: Re-add after monitor impl
    // fn screenshot_cropped(&self, x: u32, y: u32, width: u32, height: u32) -> DynamicImage;

    fn is_within(&self, x: u32, y: u32) -> bool;

    /// Finds the target image on screen. Returning a tuple of the coordinates of the top left corner of the target image.
    /// - **target** - The target image that we are looking for in the source image.
    /// - **looseness** - The percentage of difference allowed between the two pixels based on 255. 0 being no difference and 1 being a perfect match
    fn find_target(
        &self,
        method: search::SearchAlgorithm,
        target: &DynamicImage,
    ) -> Option<(u32, u32)> {
        method.find(&mut self.screenshot(), target)
    }

    // fn find_all_targets(&self, target: &DynamicImage, looseness: f32) -> Vec<(u32, u32)>;
}

#[derive(Debug, Deref, DerefMut, derive_more::From)]
pub struct Screen(xcap::Monitor);

impl Screen {
    pub fn new(id: u32) -> Option<Self> {
        ScreenSelect::Id(id).get().map(Self)
    }

    pub fn new_primary() -> Self {
        ScreenSelect::Primary.get().map(Self).unwrap()
    }
}

impl ScreenControls for Screen {
    fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    fn screenshot(&self) -> DynamicImage {
        let capture = self.capture_image().unwrap().into_raw();
        image::load_from_memory(&capture).unwrap()
    }

    fn is_within(&self, x: u32, y: u32) -> bool {
        x < self.width() && y < self.height()
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum ScreenSelect {
    #[default]
    Primary,
    Id(u32),
}

impl ScreenSelect {
    fn is_monitor(&self, monitor: &xcap::Monitor) -> bool {
        match self {
            Self::Primary => monitor.is_primary(),
            Self::Id(id) => *id == monitor.id(),
        }
    }

    pub fn get(&self) -> Option<xcap::Monitor> {
        let monitors = Monitor::all().unwrap();
        monitors.into_iter().filter(|s| self.is_monitor(s)).last()
    }
}
