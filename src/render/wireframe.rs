use crate::deformation::TetElement;
use crate::render::camera::CameraUniforms;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::BTreeSet;

/// Vertex structure for wireframe lines and interactive handle sprites.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct WireframeVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl WireframeVertex {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Camera view-projection matrix uniform layout for wireframe rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WireframeUniform {
    pub view_proj: [f32; 16],
}

/// One cage to overlay: geometry plus its edge colour.
pub struct CageGuide<'a> {
    /// Current (possibly deformed) cage vertex positions.
    pub vertices: &'a [[f32; 3]],
    /// Tetrahedral connectivity used to derive unique edges.
    pub elements: &'a [TetElement],
    /// RGBA colour for the cage edges.
    pub edge_color: [f32; 4],
}

/// Extracts unique deduplicated undirected edges $(u, v)$ with $u < v$ from tetrahedral elements.
pub fn extract_unique_edges(elements: &[TetElement]) -> Vec<[u32; 2]> {
    let mut edge_set = BTreeSet::new();

    for elem in elements {
        let idx = elem.indices;
        let pairs = [
            (idx[0], idx[1]),
            (idx[0], idx[2]),
            (idx[0], idx[3]),
            (idx[1], idx[2]),
            (idx[1], idx[3]),
            (idx[2], idx[3]),
        ];
        for (u, v) in pairs {
            let edge = if u < v { (u, v) } else { (v, u) };
            edge_set.insert(edge);
        }
    }

    edge_set.into_iter().map(|(u, v)| [u, v]).collect()
}

/// Builds the left-handed, zero-to-one depth projection used by the splat pipeline.
///
/// The splat rasterizer is +Z-forward and left-handed; using a right-handed
/// projection here would mirror the cage relative to the Gaussians it deforms.
pub fn build_wireframe_view_proj(camera: &CameraUniforms) -> Mat4 {
    let aspect = camera.viewport_width / camera.viewport_height.max(1.0);
    let fov_y = (0.5 * camera.viewport_height / camera.focal_y.max(1e-6)).atan() * 2.0;
    let proj = Mat4::perspective_lh(fov_y, aspect, camera.near_plane, 1000.0);
    proj * camera.view_mat4()
}

/// Dedicated WGPU line-list and handle marker renderer for volumetric cages.
///
/// - Topology: `wgpu::PrimitiveTopology::LineList`.
/// - Alpha blending targeting the swapchain directly.
/// - Base edges rendered in each cage's colour.
/// - Selected / dragged vertices highlighted with glowing gold diamond handles.
pub struct WireframeRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub vertex_capacity: usize,
}

impl WireframeRenderer {
    pub const SHADER_SRC: &'static str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

    /// Creates a wireframe renderer targeting the given surface swapchain format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WireframeShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WireframeBindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WireframeUniformBuffer"),
            size: std::mem::size_of::<WireframeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WireframeBindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WireframePipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("WireframeRenderPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(WireframeVertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
            vertex_buffer: None,
            vertex_capacity: 0,
        }
    }

    /// Builds the line-list vertex stream for every visible cage.
    pub fn build_vertices(
        guides: &[CageGuide<'_>],
        selected: Option<(usize, usize)>,
        camera: &CameraUniforms,
    ) -> Vec<WireframeVertex> {
        let gold = [1.0, 0.85, 0.1, 1.0];
        let handle_dim = [0.4, 0.7, 0.9, 0.5];

        let w = camera.view_matrix;
        let right = Vec3::new(w[0], w[4], w[8]).normalize_or_zero();
        let up = Vec3::new(w[1], w[5], w[9]).normalize_or_zero();

        let mut lines: Vec<WireframeVertex> = Vec::with_capacity(guides.len() * 128);

        for (cage_index, guide) in guides.iter().enumerate() {
            if guide.vertices.is_empty() || guide.elements.is_empty() {
                continue;
            }

            for [u, v] in extract_unique_edges(guide.elements) {
                let u_idx = u as usize;
                let v_idx = v as usize;
                if u_idx < guide.vertices.len() && v_idx < guide.vertices.len() {
                    lines.push(WireframeVertex::new(
                        guide.vertices[u_idx],
                        guide.edge_color,
                    ));
                    lines.push(WireframeVertex::new(
                        guide.vertices[v_idx],
                        guide.edge_color,
                    ));
                }
            }

            for (i, &v_arr) in guide.vertices.iter().enumerate() {
                let p = Vec3::from(v_arr);
                let t = camera.world_to_camera(p);
                if t.z <= camera.near_plane {
                    continue;
                }

                let is_selected = selected == Some((cage_index, i));
                let color = if is_selected { gold } else { guide.edge_color };
                let r = if is_selected {
                    0.045 * t.z
                } else {
                    0.025 * t.z
                };

                let p1 = p + right * r;
                let p2 = p + up * r;
                let p3 = p - right * r;
                let p4 = p - up * r;

                lines.push(WireframeVertex::new(p1.to_array(), color));
                lines.push(WireframeVertex::new(p2.to_array(), color));
                lines.push(WireframeVertex::new(p2.to_array(), color));
                lines.push(WireframeVertex::new(p3.to_array(), color));
                lines.push(WireframeVertex::new(p3.to_array(), color));
                lines.push(WireframeVertex::new(p4.to_array(), color));
                lines.push(WireframeVertex::new(p4.to_array(), color));
                lines.push(WireframeVertex::new(p1.to_array(), color));

                if is_selected {
                    lines.push(WireframeVertex::new(p1.to_array(), color));
                    lines.push(WireframeVertex::new(p3.to_array(), color));
                    lines.push(WireframeVertex::new(p2.to_array(), color));
                    lines.push(WireframeVertex::new(p4.to_array(), color));
                } else {
                    lines.push(WireframeVertex::new(p1.to_array(), handle_dim));
                    lines.push(WireframeVertex::new(p3.to_array(), handle_dim));
                }
            }
        }

        lines
    }

    /// Renders every cage guide into the active render pass.
    pub fn render_cages<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        guides: &[CageGuide<'_>],
        selected: Option<(usize, usize)>,
        camera: &CameraUniforms,
    ) {
        let lines = Self::build_vertices(guides, selected, camera);
        if lines.is_empty() {
            return;
        }

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&WireframeUniform {
                view_proj: build_wireframe_view_proj(camera).to_cols_array(),
            }),
        );

        let needed_verts = lines.len();
        if needed_verts > self.vertex_capacity || self.vertex_buffer.is_none() {
            let new_cap = (needed_verts * 3 / 2).max(128);
            let byte_size = (new_cap * std::mem::size_of::<WireframeVertex>()) as u64;

            self.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("WireframeVertexBuffer"),
                size: byte_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vertex_capacity = new_cap;
        }

        let Some(vb) = self.vertex_buffer.as_ref() else {
            return;
        };
        queue.write_buffer(vb, 0, bytemuck::cast_slice(&lines));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(
            0,
            vb.slice(0..(needed_verts * std::mem::size_of::<WireframeVertex>()) as u64),
        );
        render_pass.draw(0..needed_verts as u32, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_wireframe_projection_uses_left_handed_zero_to_one_depth() {
        let camera = CameraUniforms::from_look_at(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0f32.to_radians(),
            800.0,
            600.0,
            0.2,
        );

        let view_proj = build_wireframe_view_proj(&camera);

        // A point on the near plane maps to depth 0 and a point at +Z maps forward.
        let near_point = view_proj * glam::Vec4::new(0.0, 0.0, camera.near_plane - 5.0, 1.0);
        let near_ndc = near_point / near_point.w;
        assert!(
            (near_ndc.z - 0.0).abs() < 1e-3,
            "near plane must map to z=0 in a zero-to-one depth projection, got {}",
            near_ndc.z
        );

        let far_point = view_proj * glam::Vec4::new(0.0, 0.0, 1.0, 1.0);
        let far_ndc = far_point / far_point.w;
        assert!(
            far_ndc.z > 0.0 && far_ndc.z < 1.0,
            "forward points must land in (0, 1), got {}",
            far_ndc.z
        );
    }

    #[test]
    fn test_wireframe_vertex_layout_size() {
        assert_eq!(std::mem::size_of::<WireframeVertex>(), 28);
        assert_eq!(std::mem::size_of::<WireframeUniform>(), 64);
    }

    #[test]
    fn test_build_vertices_includes_edges_and_handles() {
        let camera = CameraUniforms::new(
            Mat4::IDENTITY,
            500.0,
            500.0,
            400.0,
            300.0,
            800.0,
            600.0,
            0.2,
        );
        let verts = vec![
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 2.5],
        ];
        let elements = vec![TetElement::new(0, 1, 2, 3)];
        let guide = CageGuide {
            vertices: &verts,
            elements: &elements,
            edge_color: [0.0, 0.85, 1.0, 0.45],
        };
        let lines = WireframeRenderer::build_vertices(&[guide], Some((0, 3)), &camera);
        // 6 tet edges * 2 = 12; per vertex diamond = 8 more, plus 2 handle
        // segments (or 4 for the selected cross): 3*10 + 12 = 42; total 54.
        assert_eq!(lines.len(), 54);
    }
}
