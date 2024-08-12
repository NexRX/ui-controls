use image::DynamicImage;



pub fn accelerated_template_matching(source: &DynamicImage, template: &DynamicImage) -> (f32, (u32, u32)) {
    // Or alternatively you can create the matcher first
    let mut matcher = template_matching::TemplateMatcher::new();
    matcher.match_template(
        &source.to_luma32f(),
        &template.to_luma32f(),
        template_matching::MatchTemplateMethod::SumOfSquaredDifferences,
    );

    let result = matcher.wait_for_result().unwrap();

    // Calculate min & max values
    let extremes = template_matching::find_extremes(&result);

    (extremes.max_value, extremes.max_value_location)
}

pub fn template_matching(source: &DynamicImage, template: &DynamicImage) -> (f32, (u32, u32)) {
    // Or alternatively you can create the matcher first
    let mut matcher = template_matching::TemplateMatcher::new();
    matcher.match_template(
        &source.to_luma32f(),
        &template.to_luma32f(),
        template_matching::MatchTemplateMethod::SumOfSquaredDifferences,
    );

    let result = matcher.wait_for_result().unwrap();

    // Calculate min & max values
    let extremes = template_matching::find_extremes(&result);

    (extremes.max_value, extremes.max_value_location)
}

#[cfg(test)]
mod tests {
    use template_matching::{find_extremes, match_template, MatchTemplateMethod, TemplateMatcher};

    #[test]
    fn find_button_on_screen() {
        let source = image::open("__fixtures__/screen-pow2.png")
            .unwrap()
            .to_luma32f();
        let template = image::open("__fixtures__/gb-pow2-btn.png")
            .unwrap()
            .to_luma32f();

        let result = match_template(
            &source,
            &template,
            MatchTemplateMethod::SumOfSquaredDifferences,
        );

        let time = std::time::Instant::now();
        // Or alternatively you can create the matcher first
        let mut matcher = TemplateMatcher::new();
        matcher.match_template(
            &source,
            &template,
            MatchTemplateMethod::SumOfSquaredDifferences,
        );

        let results = matcher.wait_for_result().unwrap();

        // Calculate min & max values
        let extremes = find_extremes(&result);

        println!("Time: {:?} | {:?}", time.elapsed(), extremes)
    }
}
