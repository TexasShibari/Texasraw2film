struct Params {
    highlight_burn: f32,
    log_H_ref: f32,
    red_only: f32,
};

fn safe_log10_vec3(v: vec3<f32>) -> vec3<f32> {
    let eps = 1e-6;
    return log2(max(v, vec3<f32>(eps))) / log2(10.0);
}

// ==========================================
// PASS 1: DOWNSAMPLE & FILTRATION
// ==========================================
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var smp_linear: sampler;
@group(0) @binding(2) var lowres_mask_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params_p1: Params;

@compute @workgroup_size(16, 16)
fn downsample_func(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lowres_mask_write);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let linear_color = textureSampleLevel(input_tex, smp_linear, uv, 0.0);

    // 1. Convert sample to log exposure space before thresholding
    let log_color = safe_log10_vec3(linear_color.rgb);

    let threshold_channel = select(log_color.g, log_color.r, params_p1.red_only > 0.5);
    var func_val = max(threshold_channel - params_p1.log_H_ref, 0.0);
    if (params_p1.red_only > 0.5) {
        let midtone_sigma = 0.75;
        let distance = threshold_channel - params_p1.log_H_ref;
        func_val = exp(-0.5 * distance * distance / (midtone_sigma * midtone_sigma));
    }

    textureStore(lowres_mask_write, id.xy, vec4<f32>(func_val, 0.0, 0.0, 1.0));
}

// ==========================================
// PASS 2: PRECISION SCIPY BLUR & COMPOSITION
// ==========================================
@group(0) @binding(0) var orig_highres_tex: texture_2d<f32>;
@group(0) @binding(1) var lowres_mask_read: texture_2d<f32>;
@group(0) @binding(2) var smp_bilinear: sampler;
@group(0) @binding(3) var output_highres_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params_p2: Params;

@compute @workgroup_size(16, 16)
fn final_burn(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_highres_write);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let tex_el_size = 1.0 / vec2<f32>(textureDimensions(lowres_mask_read));

    // Exact 1D SciPy Gaussian weights for sigma=3, truncate=2
    let weights = array<f32, 13>(
        0.01854402, 0.03416694, 0.05633176, 0.08310854, 0.10971929, 0.12961803,
        0.13702282, 0.12961803, 0.10971929, 0.08310854, 0.05633176, 0.03416694,
        0.01854402
    );

    var blur_accum = 0.0;

    // Outer-product 2D loop matching the radius of 6
    for (var dy: i32 = -6; dy <= 6; dy++) {
        for (var dx: i32 = -6; dx <= 6; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * tex_el_size;
            let sample_val = textureSampleLevel(lowres_mask_read, smp_bilinear, uv + offset, 0.0).r;

            // Map [-6, 6] loop indices to [0, 12] array indices
            let weight = weights[dx + 6] * weights[dy + 6];
            blur_accum += sample_val * weight;
        }
    }

    // Fetch original pixel and convert to log space
    let orig_color = textureLoad(orig_highres_tex, id.xy, 0);
    let log_orig_rgb = safe_log10_vec3(orig_color.rgb);

    // Apply highlight burn subtraction in log space
    var final_rgb = log_orig_rgb - (params_p2.highlight_burn * blur_accum);
    if (params_p2.red_only > 0.5) {
        final_rgb = vec3<f32>(
            log_orig_rgb.r - (params_p2.highlight_burn * blur_accum),
            log_orig_rgb.g,
            log_orig_rgb.b
        );
    }

    textureStore(output_highres_write, id.xy, vec4<f32>(final_rgb, orig_color.a));
}
