# UI Controls
A easy to use library for interacting with UIs and screens in general.

## State of the Project
Im primary using this as a learning experience for CPU and GPU acceleration and Computer Vision Algorythms in general. Therefor, things may not be as fast as they could be, but the aims to make make as such.

## To-Do List

### Pending

#### Screen
- [X] `size() -> (u32, u32)` - Gets the size of the screen
- [X] `screenshot() -> DynamicImage` - Takes a screenshot of the whole screen
- [X] `is_within_screen(..) -> bool` - Checks if a point is within the screen
- [X] `find_target(..) -> Option<(u32, u32)>` - Returns the top left corner of the first match of target in source
- [ ] `find_all_targets(..) -> Vec<(u32, u32)>` - Returns the top left corner of all matches of target in source

#### Screen Search
- [X] `find_target_bruteforce(..) -> Option<(u32, u32)>` - Returns the top left corner of first match of target in source using a brute force method with as little dependacies as possible
- [ ] `find_target_cv(..) -> Option<(u32, u32)>` - Returns the top left corner of first match of target in source using OpenCV methods

#### Interact
- [ ] `mouse_position() -> (u32, u32)` - Gets current mouse position
- [ ] `mouse_move_to(x: u32, y: u32)` - Moves mouse to the given coordinates
- [ ] `mouse_move_by(x: i32, y: i32)` - Moves mouse by the given offsets
- [ ] `click()` - Clicks the mouse where it is currently located
- [ ] `double_click()`  - Double clicks where it is currently located
- [ ] `click_at(x: u32, y: u32, return_mouse: bool)`  - Clicks at the given coordinates and returns mouse position
- [ ] `double_click_at(x: u32, y: u32, return_mouse: bool)` - Double clicks at the given coordinates
- [ ] `click_at_target(target: &DynamicImage, looseness: f32, offset: ClickOffset, return_mouse: bool)` - Clicks at the first match of target in the whole screen. Can use an offset with `ClickOffset::Center`, `ClickOffset::TopLeft`, etc...,  `ClickOffset::Percent(u32, u32)` or `ClickOffset::Pixels(u32, u32)`

#### Template Matching Algos
- [X] Brute Force
- [ ] CPU-Accelerated FFT (via feature flag 'cpu')
- [ ] GPU-Accelerated [TBC] (via feature flag 'gpu')
- [X] OpenCV (via feature flag 'opencv')
- [X] ImageProc / Cross Correlation Normalized (via feature flag 'imageproc')

### Extras
- [ ] Multi-screen support (beyond primary screen)
- [ ] Hardware accelerated image search

# Building

If using the feature flag `opencv`, you will ned to follow the instructions [here](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md) in order to be able to compile without errors. If this is not desirable, at time of writing, your best option for real time processing is to the brute force algorithm with max strictness (`1.0`) which will allow it to perform faster than `imageproc` but will still be fairly slow depending on the source and target/template.