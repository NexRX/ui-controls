struct Params {
    source_width: u32,
    source_height: u32,
    kernel_width: u32,
    kernel_height: u32
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> param: Params;

const WG_SIZE_X: u32 = 42;
const WG_SIZE_Y: u32 = 24;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
    let x = global.x;
    let y = global.y;

    var sum: f32 = 0.;

    let base_source_row = y * param.source_width + x;
    
    for (var trow = 0u; trow < param.kernel_height; trow = trow + 1u) {
        let source_row = base_source_row + trow * param.source_width;
        let kernel_row = param.kernel_width * trow;

        for (var tcol = 0u; tcol < param.kernel_width; tcol = tcol + 1u) {
            let source_index = source_row + tcol;
            let kernel_index = kernel_row + tcol;

            let diff = source[source_index] - kernel[kernel_index];
            sum += diff * diff;
        }
    }

    output[(param.source_width * y) + x] = sum;
}