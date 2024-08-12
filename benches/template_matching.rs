use criterion::{criterion_group, criterion_main, Criterion};
use ui_controls::matching::*;
const IMAGE_PATHS: [(&str, &str); 1] = [("__fixtures__/screen-2k.png", "__fixtures__/btn.png")];

#[cfg(feature = "cpu")]
pub fn bench_template_match_with_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Template Match (CPU) ");
    for (source, templ) in IMAGE_PATHS.into_iter() {
        let name = format!("Source: '{source}' & Template: '{templ}'");
        let source = image::open(source).unwrap();
        let template = image::open(templ).unwrap();

        group.bench_function(&name, |b| {
            b.iter(|| cpu::template_matching(&source, &template))
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
pub fn bench_template_match_with_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Template Match (GPU) ");
    for (source, templ) in IMAGE_PATHS.into_iter() {
        let name = format!("Source: '{source}' & Template: '{templ}'");
        let source = image::open(source).unwrap();
        let template = image::open(templ).unwrap();

        group.bench_function(&name, |b| {
            b.iter(|| gpu::template_matching(&source, &template))
        });
    }
    group.finish();
}

#[cfg(feature = "opencv")]
pub fn bench_template_match_with_opencv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Template Match (OpenCV) ");
    for (source, templ) in IMAGE_PATHS.into_iter() {
        let name = format!("Source: '{source}' & Template: '{templ}'");
        let source = image::open(source).unwrap();
        let template = image::open(templ).unwrap();

        group.bench_function(&name, |b| {
            b.iter(|| opencv::find_target(&source, &template, 0.99))
        });
    }
    group.finish();
}

// pub fn bench_fft2d_with_transpose(c: &mut Criterion) {
//     let mut group = c.benchmark_group("fft2d (transpose) ");
//     for n in FFT_SIZES.into_iter() {
//         let name = format!("Size: {}", n);
//         let mut v = test_vec(n * n);
//         let mut scratch = vec![Complex::default(); n * n];
//         let mut planner = FftPlanner::<f64>::new();
//         let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n);
//         group.bench_function(&name, |b| {
//             b.iter(|| fft2d_with_transpose(&mut v, &mut scratch, n, n, &fft, AXIS))
//         });
//     }
//     group.finish();
// }

criterion_group!(
    benches,
    bench_template_match_with_cpu,
    bench_template_match_with_gpu,
    bench_template_match_with_opencv,
);
criterion_main!(benches);
