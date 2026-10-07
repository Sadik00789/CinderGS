pub mod camera;
pub mod gui;
pub mod picker;

pub use camera::OrbitCamera;
pub use gui::*;
pub use picker::*;

use crate::deformation::{CageSpringSimulator, GaussianBinding, TetMesh};
use crate::render::pipeline::RenderPipelineOrchestrator;
use crate::render::rasterizer::CompositorUniforms;
use crate::scene::soa::GaussianSceneSoa;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// Main application orchestrator managing the window lifecycle, WGPU device/surface,
/// orbit camera, cage deformation picking, and egui telemetry.
pub struct CinderApp {
    // Window and WGPU Context
    pub window: Option<Arc<Window>>,
    pub surface: Option<wgpu::Surface<'static>>,
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,

    // Pipeline orchestrator
    pub orchestrator: Option<RenderPipelineOrchestrator>,

    // Scene & Cage Deformation State
    pub scene: GaussianSceneSoa,
    pub tet_mesh: Option<TetMesh>,
    pub bindings: Option<Vec<GaussianBinding>>,
    pub cage_vertices: Vec<[f32; 3]>,
    pub rest_cage_vertices: Vec<[f32; 3]>,
    pub spring_sim: Option<CageSpringSimulator>,

    // Interaction controllers
    pub camera: OrbitCamera,
    pub picker: CagePicker,
    pub gui_state: GuiState,

    // Immediate-mode UI Context & Renderer
    pub egui_ctx: egui::Context,
    pub egui_raw_input: egui::RawInput,
    pub egui_renderer: Option<egui_wgpu::Renderer>,

    // Mouse input tracking
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub middle_mouse_down: bool,
    pub last_cursor_pos: Option<[f32; 2]>,

    // Timing and telemetry
    pub start_time: Instant,
    pub last_frame_time: Instant,
    pub frame_counter: u32,
    pub fps_timer: Instant,
}

impl CinderApp {
    /// Creates a new `CinderApp` with the provided Gaussian scene and optional deformation cage.
    pub fn new(
        scene: GaussianSceneSoa,
        tet_mesh: Option<TetMesh>,
        bindings: Option<Vec<GaussianBinding>>,
    ) -> Self {
        let (cage_verts, rest_verts, spring_sim) = if let Some(mesh) = &tet_mesh {
            let sim = CageSpringSimulator::new(mesh.rest_vertices.clone());
            (mesh.rest_vertices.clone(), mesh.rest_vertices.clone(), Some(sim))
        } else {
            (Vec::new(), Vec::new(), None)
        };

        let now = Instant::now();

        Self {
            window: None,
            surface: None,
            surface_config: None,
            device: None,
            queue: None,
            orchestrator: None,

            scene,
            tet_mesh,
            bindings,
            cage_vertices: cage_verts,
            rest_cage_vertices: rest_verts,
            spring_sim,

            camera: OrbitCamera::default(),
            picker: CagePicker::new(),
            gui_state: GuiState::default(),

            egui_ctx: egui::Context::default(),
            egui_raw_input: egui::RawInput::default(),
            egui_renderer: None,

            left_mouse_down: false,
            right_mouse_down: false,
            middle_mouse_down: false,
            last_cursor_pos: None,

            start_time: now,
            last_frame_time: now,
            frame_counter: 0,
            fps_timer: now,
        }
    }

    /// Creates a demo scene with colored Gaussians bound to a 5-tet cage for interactive testing.
    pub fn with_demo_scene() -> Self {
        let mut scene = GaussianSceneSoa::with_capacity(3);
        let mut sh_r = [0.0f32; 48];
        sh_r[0] = 1.0; // Red
        let mut sh_g = [0.0f32; 48];
        sh_g[1] = 1.0; // Green
        let mut sh_b = [0.0f32; 48];
        sh_b[2] = 1.0; // Blue

        scene.push([0.0, 0.0, 2.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 0.9, &sh_r);
        scene.push([0.5, 0.3, 2.3], [0.03, 0.0, 0.0, 0.03, 0.0, 0.03], 0.8, &sh_g);
        scene.push([-0.4, -0.2, 2.5], [0.05, 0.0, 0.0, 0.05, 0.0, 0.05], 0.85, &sh_b);

        let cage = TetMesh::create_box_cage([-1.5, -1.5, 1.0], [1.5, 1.5, 3.5]);
        let pts = vec![[0.0, 0.0, 2.0], [0.5, 0.3, 2.3], [-0.4, -0.2, 2.5]];
        let bindings = cage.bind_gaussians(&pts);

        Self::new(scene, Some(cage), Some(bindings))
    }
}

impl ApplicationHandler for CinderApp {
    /// Lazy Surface Initialization: Called by OS when window surface is ready.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("CinderGS - Real-Time 3D Gaussian Splatting Engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(event_loop.create_window(window_attrs).expect("Failed to create Winit window"));
        self.window = Some(window.clone());

        // WGPU Instance & Surface Creation
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create WGPU surface from window");

        // Request Adapter & Device asynchronously (blocking once on startup)
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find compatible WGPU graphics adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("CinderGSDevice"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .expect("Failed to acquire WGPU Device and Queue");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Pipeline Orchestrator initialization
        let mut orchestrator = RenderPipelineOrchestrator::new(&device, surface_format);
        orchestrator.resize(&device, width, height);

        // Egui Renderer Initialization
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // Commit initialized state
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.orchestrator = Some(orchestrator);
        self.egui_renderer = Some(egui_renderer);

        window.request_redraw();
    }

    /// Handles window events including resizing, inputs, and redraw dispatch.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };

        let egui_wants_pointer = self.egui_ctx.wants_pointer_input();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                let width = physical_size.width.max(1);
                let height = physical_size.height.max(1);

                if let (Some(surface), Some(config), Some(device), Some(orch)) = (
                    &mut self.surface,
                    &mut self.surface_config,
                    &self.device,
                    &mut self.orchestrator,
                ) {
                    config.width = width;
                    config.height = height;
                    surface.configure(device, config);
                    orch.resize(device, width, height);
                }
                window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let is_down = state == ElementState::Pressed;
                let cur_pos = self.last_cursor_pos.unwrap_or([0.0, 0.0]);

                let egui_button = match button {
                    MouseButton::Left => egui::PointerButton::Primary,
                    MouseButton::Right => egui::PointerButton::Secondary,
                    MouseButton::Middle => egui::PointerButton::Middle,
                    _ => egui::PointerButton::Primary,
                };

                self.egui_raw_input.events.push(egui::Event::PointerButton {
                    pos: egui::pos2(cur_pos[0], cur_pos[1]),
                    button: egui_button,
                    pressed: is_down,
                    modifiers: egui::Modifiers::default(),
                });

                match button {
                    MouseButton::Left => {
                        self.left_mouse_down = is_down;
                        if !egui_wants_pointer && is_down {
                            let size = window.inner_size();
                            let camera_uniforms = self.camera.build_camera_uniforms(size.width as f32, size.height as f32);
                            let picked = self.picker.pick_vertex(cur_pos, &self.cage_vertices, &camera_uniforms);
                            self.gui_state.selected_vertex = picked;
                            self.picker.is_dragging = picked.is_some();
                            if let (Some(sim), Some(p_idx)) = (&mut self.spring_sim, picked) {
                                sim.set_pinned(Some(p_idx));
                            }
                        } else if !is_down {
                            self.picker.is_dragging = false;
                            if let Some(sim) = &mut self.spring_sim {
                                sim.set_pinned(None);
                            }
                        }
                    }
                    MouseButton::Right => {
                        self.right_mouse_down = is_down;
                    }
                    MouseButton::Middle => {
                        self.middle_mouse_down = is_down;
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = [position.x as f32, position.y as f32];
                self.egui_raw_input.events.push(egui::Event::PointerMoved(egui::pos2(new_pos[0], new_pos[1])));

                if let Some(old_pos) = self.last_cursor_pos {
                    let delta_x = new_pos[0] - old_pos[0];
                    let delta_y = new_pos[1] - old_pos[1];

                    if !egui_wants_pointer {
                        if self.left_mouse_down && self.picker.is_dragging {
                            // Eye-plane cage vertex dragging
                            let size = window.inner_size();
                            let camera_uniforms = self.camera.build_camera_uniforms(size.width as f32, size.height as f32);
                            self.picker.drag_selected([delta_x, delta_y], &mut self.cage_vertices, &camera_uniforms);
                            if let (Some(sim), Some(sel)) = (&mut self.spring_sim, self.gui_state.selected_vertex) {
                                sim.set_pinned_position(sel, self.cage_vertices[sel]);
                            }
                        } else if self.right_mouse_down {
                            // Camera orbit
                            self.camera.on_mouse_orbit(delta_x, delta_y);
                        } else if self.middle_mouse_down {
                            // Camera pan
                            self.camera.on_mouse_pan(delta_x, delta_y);
                        }
                    }
                }
                self.last_cursor_pos = Some(new_pos);
                window.request_redraw();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y * 0.05) as f32,
                };

                self.egui_raw_input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, scroll_amount * 10.0),
                    modifiers: egui::Modifiers::default(),
                });

                if !egui_wants_pointer {
                    self.camera.on_scroll(scroll_amount);
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                // Compute frame time and FPS
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;
                self.gui_state.frame_time_ms = dt * 1000.0;

                self.frame_counter += 1;
                if self.fps_timer.elapsed().as_secs_f32() >= 0.5 {
                    self.gui_state.fps = (self.frame_counter as f32) / self.fps_timer.elapsed().as_secs_f32();
                    self.frame_counter = 0;
                    self.fps_timer = Instant::now();
                }

                // Handle Jiggle Impulse button
                if self.gui_state.trigger_jiggle {
                    if let Some(sim) = &mut self.spring_sim {
                        sim.trigger_jiggle_impulse(3.5);
                    }
                    self.gui_state.trigger_jiggle = false;
                }

                // Sync physics simulator parameters
                if let Some(sim) = &mut self.spring_sim {
                    sim.stiffness = self.gui_state.spring_stiffness;
                    sim.damping = self.gui_state.spring_damping;
                }

                // Handle GUI button requests
                if self.gui_state.request_reset_cage {
                    if let Some(sim) = &mut self.spring_sim {
                        sim.reset();
                        self.cage_vertices = sim.positions_array();
                    } else {
                        self.cage_vertices = self.rest_cage_vertices.clone();
                    }
                    self.picker.clear_selection();
                    self.gui_state.selected_vertex = None;
                    self.gui_state.request_reset_cage = false;
                }
                if self.gui_state.request_reset_camera {
                    self.camera = OrbitCamera::default();
                    self.gui_state.request_reset_camera = false;
                }

                // Step physics simulation and sync cage vertices
                if let Some(sim) = &mut self.spring_sim {
                    sim.step(dt);
                    self.cage_vertices = sim.positions_array();
                }

                // Render 3DGS Scene and Blit
                let (Some(surface), Some(device), Some(queue), Some(orch), Some(config)) = (
                    &self.surface,
                    &self.device,
                    &self.queue,
                    &mut self.orchestrator,
                    &self.surface_config,
                ) else {
                    return;
                };

                // Sync execution backend
                orch.backend = self.gui_state.backend;

                let surface_texture = match surface.get_current_texture() {
                    Ok(tex) => tex,
                    Err(wgpu::SurfaceError::Lost) => {
                        surface.configure(device, config);
                        return;
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        event_loop.exit();
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {:?}", e);
                        return;
                    }
                };

                let surface_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FrameCommandEncoder"),
                });

                let size = window.inner_size();
                let width = size.width.max(1) as f32;
                let height = size.height.max(1) as f32;

                let camera_uniforms = self.camera.build_camera_uniforms(width, height);
                let compositor_uniforms = CompositorUniforms::new(
                    width as u32,
                    height as u32,
                    [0.05, 0.05, 0.07, 1.0], // Dark gray background
                    self.camera.eye_position().to_array(),
                    self.gui_state.sh_degree,
                );

                let cage_slice = if self.gui_state.show_cage && !self.cage_vertices.is_empty() {
                    Some(self.cage_vertices.as_slice())
                } else {
                    None
                };

                let stats = orch.execute_frame(
                    device,
                    queue,
                    &mut encoder,
                    &self.scene,
                    self.tet_mesh.as_ref(),
                    self.bindings.as_deref(),
                    cage_slice,
                    &camera_uniforms,
                    &compositor_uniforms,
                    &surface_view,
                    self.gui_state.show_cage,
                    self.gui_state.selected_vertex,
                );

                self.gui_state.splat_count = stats.total_gaussians;
                self.gui_state.culled_count = stats.culled_gaussians;
                self.gui_state.active_tiles = stats.active_tiles;

                // Egui pass over swapchain surface (LoadOp::Load to preserve 3DGS render)
                if let Some(egui_renderer) = &mut self.egui_renderer {
                    let scale_factor = window.scale_factor() as f32;
                    self.egui_raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(width / scale_factor, height / scale_factor),
                    ));
                    self.egui_ctx.set_pixels_per_point(scale_factor);
                    self.egui_raw_input.time = Some(self.start_time.elapsed().as_secs_f64());


                    let input = self.egui_raw_input.take();
                    let full_output = self.egui_ctx.run(input, |ctx| {
                        render_gui(ctx, &mut self.gui_state);
                    });

                    let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, scale_factor);

                    let screen_desc = egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [size.width.max(1), size.height.max(1)],
                        pixels_per_point: scale_factor,
                    };

                    for (id, delta) in &full_output.textures_delta.set {
                        egui_renderer.update_texture(device, queue, *id, delta);
                    }

                    egui_renderer.update_buffers(device, queue, &mut encoder, &clipped_primitives, &screen_desc);

                    {
                        let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("EguiRenderPass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &surface_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });

                        egui_renderer.render(&mut egui_pass, &clipped_primitives, &screen_desc);
                    }

                    for id in &full_output.textures_delta.free {
                        egui_renderer.free_texture(id);
                    }
                }

                queue.submit(std::iter::once(encoder.finish()));
                surface_texture.present();
                window.request_redraw();
            }

            _ => {}
        }
    }
}
