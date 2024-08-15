struct Params {
    source_height: u32,
    source_width: u32,
    kernel_height: u32,
    kernel_width: u32
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> param: Params;

fn squared_diff(a: f32, b: f32) -> f32 {
    let diff = a - b;
    return diff * diff;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
    let x = global.x;
    let y = global.y;

    var sum: f32 = 0.;
    for (var trow = 0u; trow < param.kernel_height; trow = trow + 1u) {
        let source_row = (y + trow) * param.source_width + x;
        let kernel_row = param.kernel_width * trow;

        for (var tcol = 0u; tcol < param.kernel_width; tcol = tcol + 1u) {
            let source_index = source_row + tcol;
            let kernel_index = kernel_row + tcol;
            sum += squared_diff(source[source_index], kernel[kernel_index]);
        }
    }

    output[(param.source_width * y) + x] = sum;
}