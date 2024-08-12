// All rights to https://www.reddit.com/r/rust/comments/14yti5s/opencl_kernel_for_template_matching/. Here temporarily for benchmarking against a future self written impl

use ocl::enums::{
    ImageChannelDataType, ImageChannelOrder, MemObjectType,
};
use ocl::{Context, Device, Image, Kernel, Program, Queue, Buffer};
use image::GrayImage;
use std::error::Error;
use std::vec;

static KERNEL_SRC: &str = r#"
    __kernel void matchTemplate_SQDIFF(
        read_only image2d_t src_image, int src_cols, int src_rows,
        read_only image2d_t templ_image, int templ_cols, int templ_rows,
        __global float* const result)
    
    {
        // x = column index, y = row index
        int x = get_global_id(0);
        int y = get_global_id(1);

        // Loop through all the pixels (done via get_global_id) and calculate the squared difference for each pixel
        if (x <= (src_cols - templ_cols) && y <= (src_rows - templ_rows))
        {    
            float sum = 0.0;

            for (int i = 0; i < templ_cols; i++)
            {
                for (int j = 0; j < templ_rows; j++)
                {
                    int2 coord_src = (int2)(x + i, y + j);
                    int2 coord_templ = (int2)(i, j);
                    float4 pixel_src = read_imagef(src_image, coord_src);
                    float4 pixel_templ = read_imagef(templ_image, coord_templ);
                    
                    float diff = pixel_templ.x - pixel_src.x;
                    sum = mad(diff, diff, sum);
                }
            }
            result[x + y * (src_cols - templ_cols + 1)] = sum;
        }

    }
"#;


/// Match a template against the original
pub fn match_template(original: &GrayImage, template: &GrayImage) -> Result<(usize, usize), Box<dyn Error>> {
    let dims_orig = original.dimensions();
    let dims_templ = template.dimensions();

    let buffer_size = (dims_orig.0 - dims_templ.0) * (dims_orig.1 - dims_templ.1);
    // In the kernel all u8 values are casted to a float from 0.0 - 1.0 --> max diff per pixel = 1.0
    // -> max sq_diff = 1² * n_rows_template * n_cols_template
    let mut vec_buffer = vec![(dims_templ.0*dims_templ.1) as f32; buffer_size as usize];


    // Build context and get device
    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()?;
    let device = context.devices()[0];

    // Create program and queue
    let queue = Queue::new(&context, device, None)?;
    let program = Program::builder()
        .src(KERNEL_SRC)
        // .devices(device)
        .build(&context)?;

    // Prepare the queue for the original image
    let orig_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Luminance)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims_orig)
        .flags(ocl::flags::MEM_READ_ONLY)
        .copy_host_slice(&original)
        .queue(queue.clone())
        .build()?;

    // Prepare the queue for the template image
    let templ_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Luminance)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims_templ)
        .flags(ocl::flags::MEM_READ_ONLY)
        .copy_host_slice(&template)
        .queue(queue.clone())
        .build()?;

    // Prepare the queue for the buffer
    let buffer = Buffer::builder()
        .queue(queue.clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(buffer_size)
        .copy_host_slice(&vec_buffer)
        .build()?;

    // Create the kernel based on the prepared arguments
    let kernel = Kernel::builder()
        .program(&program)
        .name("matchTemplate_SQDIFF")
        .queue(queue.clone())
        .global_work_size((&dims_orig.0-&dims_templ.0, &dims_orig.1-&dims_templ.1))
        .arg(&orig_image).arg(dims_orig.0 as i32).arg(dims_orig.1 as i32)
        .arg(&templ_image).arg(dims_templ.0 as i32).arg(dims_templ.1 as i32)
        .arg(&buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    // Read the data of sq_diff into the vector
    buffer.read(&mut vec_buffer)
        .enq()?;

    let mut vec_buffer = vec_buffer.into_iter().enumerate().collect::<Vec<(usize, f32)>>();
    vec_buffer.sort_by(|a, b| a.1.total_cmp(&b.1));

    let min = vec_buffer.first().ok_or_else(|| "Failed sorting")?;
    let (min_id, min_sq_diff) = (min.0, min.1);

    let elems_per_row = dims_orig.0 - dims_templ.0 + 1;
  
    
    let row_number = min_id/elems_per_row as usize;
    let col_offset = min_id%elems_per_row as usize;

    Ok((col_offset, row_number))

}