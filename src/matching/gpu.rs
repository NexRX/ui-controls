#![allow(dead_code)]
use crate::types::matrix::MatrixImage32;
use image::DynamicImage;
use log::{debug, info, trace, warn};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, Device, ShaderModule, ShaderModuleDescriptor};

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    source_width: u32,
    source_height: u32,
    kernel_width: u32,
    kernel_height: u32,
}

// TODO: This is about as optimized as we are reasonibly gonna get without design changes.
// Try and see if a OpenCL implementation could be faster and worth it.

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

enum BufferInit<'a> {
    Data(&'a MatrixImage32),
    Empty(wgpu::BufferAddress),
}

enum StoreType {
    Read,
    Write,
    Stage,
}

impl BufferInit<'_> {
    pub fn into_storage_buffer(
        self,
        device: &wgpu::Device,
        name: &str,
        ty: StoreType,
    ) -> wgpu::Buffer {
        let usage = match ty {
            StoreType::Write => {
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
            }
            StoreType::Read => BufferUsages::STORAGE,
            StoreType::Stage => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        };
        match self {
            BufferInit::Data(matrix) => {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{name} Buffer")),
                    contents: matrix.cast_slice(),
                    usage,
                })
            }
            BufferInit::Empty(size) => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{name} Buffer")),
                size,
                usage,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn into_uniform_buffer(
        self,
        device: &wgpu::Device,
        name: &str,
        write: bool,
    ) -> wgpu::Buffer {
        let usage = match write {
            true => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            false => BufferUsages::STORAGE,
        };
        match self {
            BufferInit::Data(matrix) => {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{name} Buffer")),
                    contents: matrix.cast_slice(),
                    usage,
                })
            }
            BufferInit::Empty(size) => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{name} Buffer")),
                size,
                usage,
                mapped_at_creation: false,
            }),
        }
    }
}

pub struct GPUTemplateMatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    shader: ShaderModule,
    bind_group: wgpu::BindGroup,
    source_buf: wgpu::Buffer,
    kernel_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    staged_buf: wgpu::Buffer,
    has_mapped: bool,
    pipeline: wgpu::ComputePipeline,
}

macro_rules! bind_desc {
    ($binding:literal, $ty:ident $(, $read_only:literal)?) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::$ty $( { read_only: $read_only } )?,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

fn create_bind_group(
    device: &wgpu::Device,
    source: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
    params: &Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Binding Groups"),
        layout: &device.create_bind_group_layout(&GPUTemplateMatcher::BIND_GROUP_LAYOUT),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

impl GPUTemplateMatcher {
    pub const SHADER_DESCRIPTOR: ShaderModuleDescriptor<'static> =
        wgpu::include_wgsl!("../shaders/template_matching.wgsl");
    pub const BIND_GROUP_LAYOUT: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                bind_desc!(0, Storage, true),
                bind_desc!(1, Storage, true),
                bind_desc!(2, Storage, false),
                bind_desc!(3, Uniform),
            ],
        };

    pub async fn new_defaults() -> Self {
        trace!("Creating GPU Template Matcher with defaults");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    memory_hints: wgpu::MemoryHints::Performance,
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: adapter.limits(),
                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let shader = device.create_shader_module(Self::SHADER_DESCRIPTOR);

        let pipeline = Self::create_pipeline(&device, &shader);

        use StoreType::*;
        let source_buf = BufferInit::Empty(1).into_storage_buffer(&device, "Source", Write);
        let kernel_buf = BufferInit::Empty(1).into_storage_buffer(&device, "Kernel", Write);
        let output_buf = BufferInit::Empty(1).into_storage_buffer(&device, "Output", Write);
        let staged_buf = BufferInit::Empty(1).into_storage_buffer(&device, "Staged", Stage);
        let params_buf = Params::default().into_buffer(&device);

        let bind_group =
            create_bind_group(&device, &source_buf, &kernel_buf, &output_buf, &params_buf);

        trace!("Created GPU Template Matcher");
        Self {
            device,
            queue,
            adapter,
            shader,
            bind_group,
            source_buf,
            kernel_buf,
            output_buf,
            params_buf,
            staged_buf,
            has_mapped: false,
            pipeline,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> wgpu::ComputePipeline {
        trace!("Creating pipeline");

        let bind_group_layout = device.create_bind_group_layout(&Self::BIND_GROUP_LAYOUT);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout], // Ensure this matches the layout used in the BindGroup
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Computer Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Writes all the data to buffers and recreates them if the size have changed
    /// TODO: To make this a little more flexible, we should add a option for resizing the buffers.
    /// I.e., some enum for ResizeBuffer::WhenNeeded, ResizeBuffer::Always (Smaller or larger), ResizeBuffer::AfterNSmallerFrames(u16), ResizeBuffer::Static(u64) (Always the same size, regardless)
    fn write_to_buffers(&mut self, source: &MatrixImage32, kernel: &MatrixImage32) {
        use StoreType::*;
        let is_source_resize = self.source_buf.size() != source.size() as u64;
        let is_kernel_resize = self.kernel_buf.size() != kernel.size() as u64;

        if is_source_resize {
            if self.has_mapped {
                self.source_buf.unmap();
                self.output_buf.unmap();
                // self.staged_buf.unmap();
            }
            self.source_buf =
                BufferInit::Data(source).into_storage_buffer(&self.device, "Source", Write);
            self.output_buf = BufferInit::Empty(source.size() as u64).into_storage_buffer(
                &self.device,
                "Output",
                Write,
            );
            self.staged_buf = BufferInit::Empty(source.size() as u64).into_storage_buffer(
                &self.device,
                "Staged",
                Stage,
            );
            debug!("Source, output and staged buffers resized/recreated");
        } else {
            self.queue
                .write_buffer(&self.source_buf, 0, source.cast_slice());
            trace!("Writing to source and output buffers")
        }

        if is_kernel_resize {
            if self.has_mapped {
                self.kernel_buf.unmap();
            }
            self.kernel_buf =
                BufferInit::Data(kernel).into_storage_buffer(&self.device, "Kernel", Write);
            debug!("Kernel buffer resized/recreated");
        } else {
            self.queue
                .write_buffer(&self.kernel_buf, 0, kernel.cast_slice());
            trace!("Writing to kernel buffer")
        }

        if is_source_resize || is_kernel_resize {
            if self.has_mapped {
                self.params_buf.unmap();
            }
            self.params_buf = Params::new(
                source.cols() as u32,
                source.rows() as u32,
                kernel.cols() as u32,
                kernel.rows() as u32,
            )
            .into_buffer(&self.device);

            trace!("Recreated params buf, assigned new bind group");
            self.bind_group = create_bind_group(
                &self.device,
                &self.source_buf,
                &self.kernel_buf,
                &self.output_buf,
                &self.params_buf,
            );
        }

        self.has_mapped = true;
    }

    pub fn execute_shader(&mut self, source: &MatrixImage32, kernel: &MatrixImage32) -> MatrixImage32 {
        trace!("Executing template matching shader");
        self.write_to_buffers(source, kernel);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        let (source_height, source_width) = source.dims_u32();
        {
            // NOTE: This is optimized for a 16:9 aspect ratio with a limit of 1024 WG Invocations
            const WG_SIZE_X: u32 = 42;
            const WG_SIZE_Y: u32 = 24;
            let workgroups_x = (source_width + WG_SIZE_X - 1) / WG_SIZE_X;
            let workgroups_y = (source_height + WG_SIZE_Y - 1) / WG_SIZE_Y;

            trace!("Creating a compute pass and dispatch the compute shader");
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                ..Default::default()
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            trace!("Dispatched template matching compute pass");
        }

        trace!("Coping output buffer to a staging buffer for reading");
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &self.staged_buf,
            0,
            source.size() as u64,
        );

        trace!("Submitting the command buffer to the queue");
        self.queue.submit(Some(encoder.finish()));
        self.staged_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("Failed to map result in stagging buffer");
            });

        debug!("Blocking until the GPU completes the operations and map the buffer synchronously");
        let time = std::time::Instant::now();
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        debug!(
            "Unblock from GPU operations after {:#?}, Reading results from staging buffer",
            time.elapsed()
        );
        let result: Vec<f32> = {
            let data = self.staged_buf.slice(..).get_mapped_range(); // Buffer view must be dropped unmapping staged buffer
            bytemuck::cast_slice(&data).to_vec()
        };

        self.staged_buf.unmap();

        MatrixImage32::new_raw(source_height as usize, source_width as usize, result)
    }
}

pub async fn template_matching(
    source: &DynamicImage,
    template: &DynamicImage,
    // gpu: GPUTemplateMatcher,
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
    use crate::{matching::gpu::GPUTemplateMatcher, types::matrix::MatrixImage32};
    use test_log::test;

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

        let mut gpu = GPUTemplateMatcher::new_defaults().await;

        let mut results = gpu.execute_shader(&source, &kernel);
        results = gpu.execute_shader(&source, &kernel);

        let (cords, score) = super::find_best_ssd(&results, Some(kernel.len() as f32));
        assert_eq!(cords, (1, 1), "The button should be near the top left corner. Actual XY: {cords:?}, Score: {score}, Raw Results: {results:#?}");
    }

    #[test(tokio::test)]
    async fn find_button_on_screen() {
        let source = MatrixImage32::open_image("__fixtures__/screen-2k.png").unwrap();
        let kernel = MatrixImage32::open_image("__fixtures__/btn.png").unwrap();

        // let (device, queue) = GPUInit::init_gpu(wgpu::PowerPreference::HighPerformance, None).await;
        let mut gpu = GPUTemplateMatcher::new_defaults().await;

        let time = std::time::Instant::now();
        let results = gpu.execute_shader(&source, &kernel);
        println!("Elapsed: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let results = gpu.execute_shader(&source, &kernel);
        println!("Elapsed: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let results = gpu.execute_shader(&source, &kernel);
        println!("Elapsed: {:?}", time.elapsed());

        // results.save_to_file("target/matrix.txt");

        let (cords, score) = super::find_best_ssd(&results, Some(kernel.len() as f32));
        assert_eq!(cords, (1180, 934), "Expected different location. Actual XY: {cords:?}, Score: {score}, Raw Results: <Too Large>");
    }
}
