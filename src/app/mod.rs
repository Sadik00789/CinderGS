pub mod camera;
pub mod gui;
pub mod picker;
pub mod timeline;

pub use camera::OrbitCamera;
pub use gui::*;
pub use picker::*;
pub use timeline::*;

use crate::deformation::{Cage, CageScene, TetMesh};
use crate::render::pipeline::RenderPipelineOrchestrator;
use crate::render::rasterizer::CompositorUniforms;
use crate::scene::autoframe::compute_scene_framing;
use crate::scene::soa::GaussianSceneSoa;
use glam::Vec3;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// Default colour of the first cage's wireframe.
pub const HERO_CAGE_COLOR: [f32; 4] = [0.0, 0.85, 1.0, 0.45];

/// Additional cage colours, cycled when the user adds cages.
pub const CAGE_PALETTE: [[f32; 4]; 5] = [
    [0.0, 0.85, 1.0, 0.45],
    [1.0, 0.55, 0.15, 0.45],
    [0.55, 0.85, 0.30, 0.45],
    [0.85, 0.45, 0.95, 0.45],
    [0.95, 0.85, 0.25, 0.45],
];

/// Main application orchestrator managing the window lifecycle, WGPU device/surface,
/// orbit camera, multi-cage deformation picking, timeline playback, and egui telemetry.
pub struct CinderApp {
    // Window and WGPU context
    pub window: Option<Arc<Window>>,
    pub surface: Option<wgpu::Surface<'static>>,
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
    pub init_error: Option<String>,
    pub scale_factor: f32,

    // Pipeline orchestrator
    pub orchestrator: Option<RenderPipelineOrchestrator>,

    // Scene & cage deformation state
    pub scene: GaussianSceneSoa,
    pub cages: CageScene,

    // Interaction controllers
    pub camera: OrbitCamera,
    pub picker: CagePicker,
    pub gui_state: GuiState,
    pub timeline: Timeline,
    pub last_playhead: f32,

    // Immediate-mode UI
    pub egui_ctx: egui::Context,
    pub egui_raw_input: egui::RawInput,
    pub egui_renderer: Option<egui_wgpu::Renderer>,

    // Mouse input tracking (physical pixels)
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
    /// Creates a new `CinderApp` with the provided Gaussian scene and cage set.
    pub fn new(scene: GaussianSceneSoa, cages: CageScene) -> Self {
        let now = Instant::now();

        Self {
            window: None,
            surface: None,
            surface_config: None,
            device: None,
            queue: None,
            init_error: None,
            scale_factor: 1.0,

            orchestrator: None,

            scene,
            cages,

            camera: OrbitCamera::default(),
            picker: CagePicker::new(),
            gui_state: GuiState::default(),
            timeline: Timeline::new(),
            last_playhead: 0.0,

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

    /// Creates an app with a single cage bound to the scene's auto-framed hero box.
    pub fn from_single_cage(scene: GaussianSceneSoa, mesh: TetMesh) -> Self {
        let framing = compute_scene_framing(&scene);
        let cages = CageScene::auto_bind(
            vec![Cage::new("hero", mesh, HERO_CAGE_COLOR)],
            &scene.positions,
        );
        let mut app = Self::new(scene, cages);
        app.camera = OrbitCamera::new(Vec3::from(framing.center), framing.camera_distance);
        app
    }

    /// Creates a demo scene with colored Gaussians bound to a 5-tet cage.
    pub fn with_demo_scene() -> Self {
        let mut scene = GaussianSceneSoa::with_capacity(3);
        let mut sh_r = [0.0f32; 48];
        sh_r[0] = 1.0; // Red
        let mut sh_g = [0.0f32; 48];
        sh_g[1] = 1.0; // Green
        let mut sh_b = [0.0f32; 48];
        sh_b[2] = 1.0; // Blue

        scene.push(
            [0.0, 0.0, 2.0],
            [0.04, 0.0, 0.0, 0.04, 0.0, 0.04],
            0.9,
            &sh_r,
        );
        scene.push(
            [0.5, 0.3, 2.3],
            [0.03, 0.0, 0.0, 0.03, 0.0, 0.03],
            0.8,
            &sh_g,
        );
        scene.push(
            [-0.4, -0.2, 2.5],
            [0.05, 0.0, 0.0, 0.05, 0.0, 0.05],
            0.85,
            &sh_b,
        );

        let cage = TetMesh::create_box_cage([-1.5, -1.5, 1.0], [1.5, 1.5, 3.5]);
        let bindings = cage.bind_gaussians(&[[0.0, 0.0, 2.0], [0.5, 0.3, 2.3], [-0.4, -0.2, 2.5]]);
        debug_assert_eq!(bindings.len(), 3);

        let cages = CageScene::auto_bind(
            vec![Cage::new("hero", cage, HERO_CAGE_COLOR)],
            &scene.positions,
        );
        let mut app = Self::new(scene, cages);
        app.camera = OrbitCamera::new(Vec3::new(0.0, 0.0, 2.0), 4.0);
        app
    }

    /// Rebuilds bindings after the cage topology changed.
    fn rebind_cages(&mut self, cages: Vec<Cage>) {
        self.cages = CageScene::auto_bind(cages, &self.scene.positions);
        self.gui_state.selected = None;
        self.picker.clear_selection();
        let valid = self.gui_state.active_cage < self.cages.cages.len();
        if !valid {
            self.gui_state.active_cage = self.cages.cages.len().saturating_sub(1);
        }
        if let Some(orchestrator) = &mut self.orchestrator {
            orchestrator.sort_cache.invalidate();
        }
    }

    /// Adds an independent cage offset from the auto-framed hero box.
    fn add_cage(&mut self) {
        let framing = compute_scene_framing(&self.scene);
        let extent = framing.hero_extent();
        let index = self.cages.cages.len();
        let offset = extent[0].max(0.25) * 1.25 * index as f32;

        let min = [
            framing.hero_min[0] + offset,
            framing.hero_min[1],
            framing.hero_min[2],
        ];
        let max = [
            framing.hero_max[0] + offset,
            framing.hero_max[1],
            framing.hero_max[2],
        ];

        let color = CAGE_PALETTE[index % CAGE_PALETTE.len()];
        let cage = Cage::new(
            format!("cage{}", index + 1),
            TetMesh::create_box_cage(min, max),
            color,
        );

        let mut cages = std::mem::take(&mut self.cages);
        cages.cages.push(cage);
        self.rebind_cages(cages.cages);
        self.gui_state.active_cage = index;
    }

    /// Removes the last cage and rebinds the remaining set.
    fn remove_cage(&mut self) {
        let mut cages = std::mem::take(&mut self.cages);
        cages.cages.pop();
        self.rebind_cages(cages.cages);
    }

    /// Captures the current camera and cage offsets into a keyframe at the playhead.
    fn bake_keyframe(&mut self) {
        let mut cage_offsets = Vec::new();
        for cage in &self.cages.cages {
            for (current, rest) in cage
                .simulator
                .current_positions
                .iter()
                .zip(cage.simulator.rest_positions.iter())
            {
                let offset = *current - *rest;
                cage_offsets.push([offset.x, offset.y, offset.z]);
            }
        }

        self.timeline.insert(Keyframe {
            time: self.timeline.playhead.max(0.0),
            camera: CameraPose {
                target: self.camera.target.to_array(),
                distance: self.camera.distance,
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
            },
            cage_offsets,
        });
    }

    /// Applies a sampled timeline pose to the camera and every cage.
    fn apply_timeline(&mut self, sample: &Keyframe) {
        self.camera.target = Vec3::from(sample.camera.target);
        self.camera.distance = sample.camera.distance.max(0.05);
        self.camera.yaw = sample.camera.yaw;
        self.camera.pitch = sample
            .camera
            .pitch
            .clamp(OrbitCamera::PITCH_MIN, OrbitCamera::PITCH_MAX);

        let mut index = 0usize;
        for cage in &mut self.cages.cages {
            for (vertex, rest) in cage
                .simulator
                .current_positions
                .iter_mut()
                .zip(cage.simulator.rest_positions.iter())
            {
                if let Some(offset) = sample.cage_offsets.get(index) {
                    *vertex = *rest + Vec3::from(*offset);
                }
                index += 1;
            }
            cage.simulator.velocities.fill(Vec3::ZERO);
        }
    }

    /// Reconfigures the surface and offscreen target for a new physical size.
    fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        let Some(surface) = self.surface.as_ref() else {
            return;
        };
        let Some(device) = self.device.as_ref() else {
            return;
        };
        let Some(config) = self.surface_config.as_mut() else {
            return;
        };

        config.width = width;
        config.height = height;
        surface.configure(device, config);

        if let Some(orchestrator) = self.orchestrator.as_mut() {
            orchestrator.resize(device, width, height);
        }
    }
}

/// Runs the egui overlay and records its render pass.
///
/// Free function rather than a method so the caller can hold the surface, device,
/// and queue borrows alive while the GUI mutably borrows its own state.
#[allow(clippy::too_many_arguments)]
fn run_egui(
    egui_ctx: &egui::Context,
    egui_raw_input: &mut egui::RawInput,
    renderer: &mut Option<egui_wgpu::Renderer>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    surface_view: &wgpu::TextureView,
    scale_factor: f32,
    start_time: Instant,
    size_in_pixels: [u32; 2],
    gui_state: &mut GuiState,
    cages: &mut CageScene,
    timeline: &mut Timeline,
) -> Vec<wgpu::CommandBuffer> {
    let scale_factor = scale_factor.max(0.1);
    egui_raw_input.screen_rect = Some(egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(
            size_in_pixels[0] as f32 / scale_factor,
            size_in_pixels[1] as f32 / scale_factor,
        ),
    ));
    egui_ctx.set_pixels_per_point(scale_factor);
    egui_raw_input.time = Some(start_time.elapsed().as_secs_f64());

    let input = egui_raw_input.take();
    let full_output = egui_ctx.run_pass(input, |ctx| render_gui(ctx, gui_state, cages, timeline));

    let clipped = egui_ctx.tessellate(full_output.shapes, scale_factor);

    let screen_desc = egui_wgpu::ScreenDescriptor {
        size_in_pixels,
        pixels_per_point: scale_factor,
    };

    let Some(renderer) = renderer.as_mut() else {
        return Vec::new();
    };

    for (id, deltas) in &full_output.textures_delta.set {
        for delta in deltas {
            renderer.update_texture(device, queue, *id, delta);
        }
    }

    let user_cmd_bufs = renderer.update_buffers(device, queue, encoder, &clipped, &screen_desc);

    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("EguiRenderPass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        let mut pass = pass.forget_lifetime();
        renderer.render(&mut pass, &clipped, &screen_desc);
    }

    for id in &full_output.textures_delta.free {
        renderer.free_texture(id);
    }

    user_cmd_bufs
}

impl ApplicationHandler for CinderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if self.init_error.is_some() {
            event_loop.exit();
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("CinderGS - Real-Time 3D Gaussian Splatting Engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = match event_loop.create_window(window_attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                self.init_error = Some(format!("failed to create window: {err}"));
                eprintln!("error: failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        self.scale_factor = window.scale_factor() as f32;
        self.egui_ctx.set_pixels_per_point(self.scale_factor);
        self.window = Some(window.clone());

        let instance = wgpu::Instance::default();
        let surface = match instance.create_surface(window.clone()) {
            Ok(surface) => surface,
            Err(err) => {
                self.init_error = Some(format!("failed to create WGPU surface: {err}"));
                eprintln!("error: failed to create WGPU surface: {err}");
                event_loop.exit();
                return;
            }
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
        })
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: true,
                apply_limit_buckets: false,
            }))
        });

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(err) => {
                self.init_error = Some(format!("failed to find a WGPU adapter: {err}"));
                eprintln!("error: failed to find a WGPU adapter: {err}");
                event_loop.exit();
                return;
            }
        };

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("CinderGSDevice"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                ..Default::default()
            })) {
                Ok(device) => device,
                Err(err) => {
                    self.init_error = Some(format!("failed to acquire WGPU device: {err}"));
                    eprintln!("error: failed to acquire WGPU device: {err}");
                    event_loop.exit();
                    return;
                }
            };

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &surface_config);

        let mut orchestrator = RenderPipelineOrchestrator::new(&device, surface_format);
        orchestrator.resize(&device, width, height);

        match orchestrator.attach_shared_gpu(
            instance,
            adapter,
            device.clone(),
            queue.clone(),
            adapter_info.backend,
        ) {
            Ok(()) => {
                self.gui_state.gpu_description = orchestrator.compute_description();
            }
            Err(err) => {
                eprintln!(
                    "warning: CubeCL could not share the presentation device ({err}); \
                     compute will acquire its own device on demand"
                );
            }
        }

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
        self.device = Some(device);
        self.queue = Some(queue);
        self.orchestrator = Some(orchestrator);
        self.egui_renderer = Some(egui_renderer);

        println!(
            "CinderGS window ready: {} ({:?}), {}x{} @ {:.2} px/pt",
            adapter_info.name, adapter_info.backend, width, height, self.scale_factor
        );

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let scale = self.scale_factor.max(0.1);
        let egui_wants_pointer = self.egui_ctx.egui_wants_pointer_input();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                self.resize(physical_size.width, physical_size.height);
                window.request_redraw();
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // Cursor positions from winit are physical pixels; egui works in
                // logical points. Keep both in sync when the monitor changes.
                self.scale_factor = scale_factor as f32;
                self.egui_ctx.set_pixels_per_point(self.scale_factor);
                let size = window.inner_size();
                self.resize(size.width, size.height);
                window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let is_down = state == ElementState::Pressed;
                let physical_pos = self.last_cursor_pos.unwrap_or([0.0, 0.0]);
                let logical_pos = egui::pos2(physical_pos[0] / scale, physical_pos[1] / scale);

                let egui_button = match button {
                    MouseButton::Left => egui::PointerButton::Primary,
                    MouseButton::Right => egui::PointerButton::Secondary,
                    MouseButton::Middle => egui::PointerButton::Middle,
                    _ => egui::PointerButton::Primary,
                };

                self.egui_raw_input.events.push(egui::Event::PointerButton {
                    pos: logical_pos,
                    button: egui_button,
                    pressed: is_down,
                    modifiers: egui::Modifiers::default(),
                });

                match button {
                    MouseButton::Left => {
                        self.left_mouse_down = is_down;
                        if !egui_wants_pointer && is_down {
                            let size = window.inner_size();
                            let camera = self.camera.build_camera_uniforms(
                                size.width.max(1) as f32,
                                size.height.max(1) as f32,
                            );
                            let cage_vertices: Vec<Vec<[f32; 3]>> = self.cages.deformed_vertices();
                            let picked =
                                self.picker
                                    .pick_cage_vertex(physical_pos, &cage_vertices, &camera);
                            self.gui_state.selected = picked;
                            self.picker.is_dragging = picked.is_some();
                            if let Some((cage_index, vertex_index)) = picked {
                                self.gui_state.active_cage = cage_index;
                                if let Some(cage) = self.cages.cages.get_mut(cage_index) {
                                    cage.simulator.set_pinned(Some(vertex_index));
                                }
                            }
                        } else if !is_down {
                            self.picker.is_dragging = false;
                            for cage in &mut self.cages.cages {
                                cage.simulator.set_pinned(None);
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
                let physical = [position.x as f32, position.y as f32];
                self.egui_raw_input
                    .events
                    .push(egui::Event::PointerMoved(egui::pos2(
                        physical[0] / scale,
                        physical[1] / scale,
                    )));

                if let Some(old_pos) = self.last_cursor_pos {
                    let delta_x = physical[0] - old_pos[0];
                    let delta_y = physical[1] - old_pos[1];

                    if !egui_wants_pointer {
                        if self.left_mouse_down && self.picker.is_dragging {
                            let size = window.inner_size();
                            let camera = self.camera.build_camera_uniforms(
                                size.width.max(1) as f32,
                                size.height.max(1) as f32,
                            );
                            if let Some(selected) = self.gui_state.selected {
                                let mut cage_vertices = self.cages.deformed_vertices();
                                if self.picker.drag_cage_vertex(
                                    [delta_x, delta_y],
                                    &mut cage_vertices,
                                    &camera,
                                    selected,
                                ) {
                                    let (cage_index, vertex_index) = selected;
                                    let position = cage_vertices
                                        .get(cage_index)
                                        .and_then(|vertices| vertices.get(vertex_index))
                                        .copied();
                                    if let (Some(cage), Some(position)) =
                                        (self.cages.cages.get_mut(cage_index), position)
                                    {
                                        cage.simulator.set_pinned_position(vertex_index, position);
                                    }
                                }
                            }
                        } else if self.right_mouse_down {
                            self.camera.on_mouse_orbit(delta_x, delta_y);
                        } else if self.middle_mouse_down {
                            self.camera.on_mouse_pan(delta_x, delta_y);
                        }
                    }
                }
                self.last_cursor_pos = Some(physical);
                window.request_redraw();
            }

            WindowEvent::CursorLeft { .. } => {
                self.egui_raw_input.events.push(egui::Event::PointerGone);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y * 0.05) as f32,
                };

                self.egui_raw_input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, scroll_amount * 10.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::default(),
                });

                if !egui_wants_pointer {
                    self.camera.on_scroll(scroll_amount);
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;
                self.gui_state.frame_time_ms = dt * 1000.0;

                self.frame_counter += 1;
                if self.fps_timer.elapsed().as_secs_f32() >= 0.5 {
                    self.gui_state.fps =
                        (self.frame_counter as f32) / self.fps_timer.elapsed().as_secs_f32();
                    self.frame_counter = 0;
                    self.fps_timer = Instant::now();
                }

                // GUI requests that change cage topology.
                if self.gui_state.add_cage {
                    self.add_cage();
                    self.gui_state.add_cage = false;
                }
                if self.gui_state.remove_cage {
                    self.remove_cage();
                    self.gui_state.remove_cage = false;
                }
                if self.gui_state.trigger_jiggle {
                    if let Some(cage) = self.cages.cages.get_mut(self.gui_state.active_cage) {
                        cage.jiggle(3.5);
                    }
                    self.gui_state.trigger_jiggle = false;
                }
                if self.gui_state.request_reset_cage {
                    if let Some(cage) = self.cages.cages.get_mut(self.gui_state.active_cage) {
                        cage.reset();
                    }
                    self.gui_state.selected = None;
                    self.picker.clear_selection();
                    self.gui_state.request_reset_cage = false;
                }
                if self.gui_state.request_reset_camera {
                    self.camera = OrbitCamera::default();
                    self.gui_state.request_reset_camera = false;
                }
                if self.gui_state.bake_keyframe {
                    self.bake_keyframe();
                    self.gui_state.bake_keyframe = false;
                }
                if self.gui_state.clear_timeline {
                    self.timeline.clear();
                    self.gui_state.clear_timeline = false;
                }

                // Timeline playback / scrubbing takes precedence over the spring
                // simulation; otherwise the cages run their own physics.
                self.timeline.advance(dt);
                let playhead_changed = (self.timeline.playhead - self.last_playhead).abs() > 1e-6;
                self.last_playhead = self.timeline.playhead;

                if self.timeline.is_active() && (self.timeline.playing || playhead_changed) {
                    if let Some(sample) = self.timeline.sample_current() {
                        self.apply_timeline(&sample);
                    }
                } else {
                    self.cages.step(dt);
                }

                let Some(surface) = self.surface.as_ref() else {
                    return;
                };
                let Some(device) = self.device.as_ref() else {
                    return;
                };
                let Some(queue) = self.queue.as_ref() else {
                    return;
                };
                let Some(config) = self.surface_config.as_ref() else {
                    return;
                };
                let Some(orchestrator) = self.orchestrator.as_mut() else {
                    return;
                };

                orchestrator.backend = self.gui_state.backend;

                let surface_texture = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        surface.configure(device, config);
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Timeout
                    | wgpu::CurrentSurfaceTexture::Occluded => return,
                    wgpu::CurrentSurfaceTexture::Validation => {
                        eprintln!("surface validation error while acquiring the next frame");
                        return;
                    }
                };

                let surface_view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("FrameCommandEncoder"),
                });

                // The offscreen target is the single source of truth for the
                // viewport size used by projection, binning, upload, and blit.
                let width = orchestrator.offscreen_width.max(1);
                let height = orchestrator.offscreen_height.max(1);

                let camera_uniforms = self
                    .camera
                    .build_camera_uniforms(width as f32, height as f32);
                let compositor_uniforms = CompositorUniforms::new(
                    width,
                    height,
                    [0.05, 0.05, 0.07, 1.0],
                    self.camera.eye_position().to_array(),
                    self.gui_state.sh_degree,
                );

                let stats = orchestrator.execute_frame(
                    device,
                    queue,
                    &mut encoder,
                    &self.scene,
                    Some(&self.cages),
                    &camera_uniforms,
                    &compositor_uniforms,
                    &surface_view,
                    self.gui_state.show_cage,
                    self.gui_state.selected,
                );

                self.gui_state.splat_count = stats.total_gaussians;
                self.gui_state.culled_count = stats.culled_gaussians;
                self.gui_state.active_tiles = stats.active_tiles;
                self.gui_state.sort_reused = stats.sort_reused;
                if let Some(description) = orchestrator.compute_description() {
                    self.gui_state.gpu_description = Some(description);
                }
                self.gui_state.gpu_error = orchestrator.last_backend_error.clone();

                let user_cmd_bufs = run_egui(
                    &self.egui_ctx,
                    &mut self.egui_raw_input,
                    &mut self.egui_renderer,
                    device,
                    queue,
                    &mut encoder,
                    &surface_view,
                    self.scale_factor,
                    self.start_time,
                    [width, height],
                    &mut self.gui_state,
                    &mut self.cages,
                    &mut self.timeline,
                );

                queue.submit(user_cmd_bufs);
                queue.submit(std::iter::once(encoder.finish()));
                queue.present(surface_texture);
                window.request_redraw();
            }

            _ => {}
        }
    }
}
