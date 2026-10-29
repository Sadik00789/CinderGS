//! Fullscreen blit pipeline.
//!
//! The offscreen compositor target uses [`OFFSCREEN_FORMAT`], an sRGB format.
//! Compositor values are therefore stored once in display-referred encoding; the
//! blit samples them (hardware decodes sRGB to linear) and the swapchain encodes
//! exactly once on presentation. When a surface only offers a non-sRGB format the
//! fragment shader performs that single encode itself, so every target produces
//! the same bytes.

/// Offscreen compositor texture format (sRGB, 8-bit).
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Fullscreen blit pipeline using a single oversized procedural triangle.
pub struct BlitPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub target_format: wgpu::TextureFormat,
}

impl BlitPipeline {
    const VERTEX_SRC: &'static str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
"#;

    /// Fragment shader body for an sRGB target (no software encode).
    const FRAGMENT_SRGB: &'static str = r#"
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
"#;

    /// Fragment shader body for a linear target (explicit sRGB encode).
    const FRAGMENT_LINEAR: &'static str = r#"
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return 12.92 * c;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(t_diffuse, s_diffuse, in.uv);
    return vec4<f32>(
        linear_to_srgb(sampled.r),
        linear_to_srgb(sampled.g),
        linear_to_srgb(sampled.b),
        sampled.a,
    );
}
"#;

    /// Creates a new fullscreen blit render pipeline targeting the given format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let source = format!(
            "{}{}",
            Self::VERTEX_SRC,
            if target_format.is_srgb() {
                Self::FRAGMENT_SRGB
            } else {
                Self::FRAGMENT_LINEAR
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BlitShader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BlitBindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BlitPipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BlitRenderPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BlitSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // Nearest keeps the copy pixel-exact, which matters for both the
            // golden-image tests and headless frame export.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            target_format,
        }
    }

    /// Creates a bind group binding the offscreen texture view and sampler.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlitBindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Records the fullscreen blit draw command into an active render pass.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offscreen_format_is_srgb() {
        assert!(OFFSCREEN_FORMAT.is_srgb());
        assert_eq!(OFFSCREEN_FORMAT, wgpu::TextureFormat::Rgba8UnormSrgb);
    }
}
