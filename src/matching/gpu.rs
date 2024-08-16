use std::{
    cell::OnceCell,
    sync::{Arc, Mutex},
};

use image::{DynamicImage, GenericImageView};
use rayon::iter::{ParallelBridge, ParallelIterator};
use wgpu::{util::DeviceExt, BufferUsages, Device, Queue, ShaderModule, ShaderModuleDescriptor};

use crate::types::matrix::{Matrix, MatrixImage, MatrixImage32};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    source_width: u32,
    source_height: u32,
    kernel_width: u32,
    kernel_height: u32,
}

impl Params {
    fn new(source_width: u32, source_height: u32, kernel_width: u32, kernel_height: u32) -> Self {
        Self {
            source_width,
            source_height,
            kernel_width,
            kernel_height,
        }
    }

    fn into_buffer(self, device: &Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[self]),
            usage: BufferUsages::UNIFORM,
        })
    }
}

struct GPUTemplateMatchState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    shader_module: ShaderModule,
    // pipeline: wgpu::ComputePipeline,
}

impl GPUTemplateMatchState {
    const SHADER_MODULE: ShaderModuleDescriptor<'static> =
        wgpu::include_wgsl!("../shaders/template_matching.wgsl");

    async fn new_defaults(source: &MatrixImage32, kernel: &MatrixImage32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let shader_module = device.create_shader_module(Self::SHADER_MODULE);

        Self {
            device,
            queue,
            adapter,
            shader_module,
        }
    }

    pub fn create_pipeline(
        &self,
        source: &MatrixImage32,
        kernel: &MatrixImage32,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup, wgpu::Buffer) {
        let source_buf =
            create_matrix_buffer("Source", &self.device, BufferUsages::STORAGE, source);
        let kernel_buf =
            create_matrix_buffer("Kernel", &self.device, BufferUsages::STORAGE, kernel);

        let (source_height, source_width) = source.dims_u32();
        let (kernel_height, kernel_width) = kernel.dims_u32();
        assert_eq!(source.len(), (source_width * source_height) as usize);
        assert_eq!(kernel.len(), (kernel_height * kernel_width) as usize);

        let output = MatrixImage32::new_zeros(source_height as usize, source_width as usize);
        let output_buf = create_matrix_buffer(
            "Output",
            &self.device,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            &output,
        );
        assert_eq!(source_buf.size(), output_buf.size());

        let params_buf = Params::new(source_width, source_height, kernel_width, kernel_height)
            .into_buffer(&self.device);

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout =
            &self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind Group Layout"),
                    entries: &[
                        create_binding_group(
                            0,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        create_binding_group(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        create_binding_group(
                            2,
                            wgpu::BufferBindingType::Storage { read_only: false },
                        ),
                        create_binding_group(3, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binding Groups"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Create the pipeline layout using the bind group layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[bind_group_layout], // Ensure this matches the layout used in the BindGroup
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Computer Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &self.shader_module,
                    entry_point: "main",
                    compilation_options: Default::default(),
                    cache: None,
                });

        (compute_pipeline, bind_group, output_buf)
    }
}

pub async fn template_matching(
    source: &DynamicImage,
    template: &DynamicImage,
    gpu: GPUTemplateMatchState,
    // TODO: confidence_threshold: f32,
) -> ((usize, usize), f32) {
    let source = MatrixImage32::from(source);
    let kernel = MatrixImage32::from(template);

    todo!();

    // let (device, queue) = GPUInit::init_gpu(wgpu::PowerPreference::HighPerformance, None).await;

    // let data = execute_shader(&device, &queue, &source, &kernel);
    // let result = find_best_ssd(&data, Some(kernel.len() as f32));

    // #[allow(clippy::let_and_return)]
    // result
}

fn execute_shader(
    gpu: &GPUTemplateMatchState,
    source: &MatrixImage32,
    kernel: &MatrixImage32,
) -> MatrixImage32 {
    let (compute_pipeline, bind_group, output_buf) = gpu.create_pipeline(source, kernel);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    let (source_height, source_width) = source.dims_u32();
    {
        const WG_SIZE_XY: u32 = 16;
        let workgroups_x = (source_width + WG_SIZE_XY - 1) / WG_SIZE_XY;
        let workgroups_y = (source_height + WG_SIZE_XY - 1) / WG_SIZE_XY;

        // Create a compute pass and dispatch the compute shader
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            ..Default::default()
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // Copy output buffer to a staging buffer for reading
    let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: source.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, source.size() as u64);

    // Submit the command buffer to the queue
    gpu.queue.submit(Some(encoder.finish()));

    // Block until the GPU completes the operations and map the buffer synchronously
    staging_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| {
            result.expect("Failed to map result in stagging buffer");
        });
    gpu.device.poll(wgpu::Maintain::Wait);

    // Read the data
    let data = staging_buf.slice(..).get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

    // Unmap the buffer
    drop(data);
    staging_buf.unmap();

    MatrixImage32::new_raw(source_height as usize, source_width as usize, result)
}

fn create_matrix_buffer(
    name: &str,
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    data: &MatrixImage32,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{name} Buffer")),
        contents: data.cast_slice(),
        usage,
    })
}

fn create_binding_group(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Returns the  X,Y cordinates or in otherwords the Column,Row of the best match detirmined by SSD
/// If you want a percentage based score, give the Kernal length which should be equal to `kernel.rows() * source.cols()`. Otherwise, give none for just the raw score.
fn find_best_ssd(data: &MatrixImage32, kernel_len: Option<f32>) -> ((usize, usize), f32) {
    let ((y, x), score) = data
        .enumerate_element()
        .reduce(|a, b| {
            if a.1 < b.1 {
                return a;
            }
            b
        })
        .unwrap();

    let score = match kernel_len {
        Some(kernel_len) => {
            let max = kernel_len * f32::MAX;
            let normalized_score = if *score != 0. { score / max } else { 0. };
            normalized_score * 100.
        }
        None => *score,
    };

    ((x, y), score)
}

#[cfg(test)]
mod tests {
    use super::execute_shader;
    use crate::{matching::gpu::GPUTemplateMatchState, types::matrix::MatrixImage32};

    #[tokio::test]
    async fn find_basic() {
        let source = MatrixImage32::new_raw(
            4,
            4,
            vec![
                0.0, 0.5, 1.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, //
                0.0, 0.5, 1.0, 0.0, //
            ],
        );
        let kernel = MatrixImage32::new_raw(
            2,
            2,
            vec![
                1.0, 1.0, //
                1.0, 1.0, //
            ],
        );

        //  let (device, queue) = GPUInit::init_gpu(wgpu::PowerPreference::HighPerformance, None).await;
        let gpu = GPUTemplateMatchState::new_defaults(&source, &kernel).await;

        let results = execute_shader(&gpu, &source, &kernel);

        let (cords, score) = super::find_best_ssd(&results, Some(kernel.len() as f32));
        assert_eq!(cords, (1, 1), "The button should be near the top left corner. Actual XY: {cords:?}, Score: {score}, Raw Results: {results:#?}");
    }

    #[tokio::test]
    async fn find_button_on_screen() {
        let source = MatrixImage32::open_image("__fixtures__/screen-2k.png").unwrap();
        let kernel = MatrixImage32::open_image("__fixtures__/btn.png").unwrap();

        // let (device, queue) = GPUInit::init_gpu(wgpu::PowerPreference::HighPerformance, None).await;
        let gpu = GPUTemplateMatchState::new_defaults(&source, &kernel).await;

        let time = std::time::Instant::now();
        let results = execute_shader(&gpu, &source, &kernel);
        println!("Elapsed: {:?}", time.elapsed());

        // results.save_to_file("target/matrix.txt");

        let (cords, score) = super::find_best_ssd(&results, Some(kernel.len() as f32));
        assert_eq!(cords, (1180, 934), "Expected different location. Actual XY: {cords:?}, Score: {score}, Raw Results: <Too Large>");
    }
}
