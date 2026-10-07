use crate::render::ExecutionBackend;

/// Immediate-mode UI state and performance telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct GuiState {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub splat_count: usize,
    pub culled_count: usize,
    pub active_tiles: usize,
    pub selected_vertex: Option<usize>,
    pub sh_degree: u32,
    pub show_cage: bool,
    pub backend: ExecutionBackend,
    pub spring_stiffness: f32,
    pub spring_damping: f32,
    pub trigger_jiggle: bool,
    pub request_reset_camera: bool,
    pub request_reset_cage: bool,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            fps: 60.0,
            frame_time_ms: 16.6,
            splat_count: 0,
            culled_count: 0,
            active_tiles: 0,
            selected_vertex: None,
            sh_degree: 3,
            show_cage: true,
            backend: ExecutionBackend::CpuReference,
            spring_stiffness: 180.0,
            spring_damping: 0.92,
            trigger_jiggle: false,
            request_reset_camera: false,
            request_reset_cage: false,
        }
    }
}

/// Renders the immediate-mode HUD overlay using egui.
pub fn render_gui(ctx: &egui::Context, state: &mut GuiState) {
    egui::Window::new("CinderGS Engine Telemetry")
        .default_pos(egui::pos2(16.0, 16.0))
        .default_size(egui::vec2(300.0, 440.0))
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Performance & Pipeline");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("FPS:");
                ui.strong(format!("{:.1}", state.fps));
                ui.label("Frame Time:");
                ui.strong(format!("{:.2} ms", state.frame_time_ms));
            });

            ui.separator();
            ui.label(format!("Total Gaussians: {}", state.splat_count));
            ui.label(format!("Culled Gaussians: {}", state.culled_count));
            let visible = state.splat_count.saturating_sub(state.culled_count);
            ui.label(format!("Rendered Splats: {}", visible));
            ui.label(format!("Active 16x16 Tiles: {}", state.active_tiles));

            ui.separator();
            ui.heading("Execution Backend");
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut state.backend,
                    ExecutionBackend::CpuReference,
                    "CPU Reference",
                );
                ui.selectable_value(
                    &mut state.backend,
                    ExecutionBackend::CubeClGpu,
                    "CubeCL GPU",
                );
            });

            ui.separator();
            ui.heading("Controls & Appearance");

            ui.horizontal(|ui| {
                ui.label("SH Degree:");
                ui.add(egui::Slider::new(&mut state.sh_degree, 0..=3));
            });

            ui.checkbox(&mut state.show_cage, "Show Cage Wireframe");

            ui.separator();
            ui.heading("Cage Elasticity & Dynamics");
            ui.horizontal(|ui| {
                ui.label("Stiffness:");
                ui.add(egui::Slider::new(&mut state.spring_stiffness, 10.0..=500.0).text("N/m"));
            });
            ui.horizontal(|ui| {
                ui.label("Damping:");
                ui.add(egui::Slider::new(&mut state.spring_damping, 0.50..=0.99));
            });
            if ui.button("Trigger Jiggle Impulse").clicked() {
                state.trigger_jiggle = true;
            }

            ui.separator();
            ui.heading("Cage Interaction");
            if let Some(idx) = state.selected_vertex {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 255, 100),
                    format!("Selected Vertex: #{}", idx),
                );
                ui.label("Drag with Left Mouse Button to deform");
            } else {
                ui.label("Click cage vertex handle to pick and deform");
            }

            ui.horizontal(|ui| {
                if ui.button("Reset Cage").clicked() {
                    state.request_reset_cage = true;
                }
                if ui.button("Reset Camera").clicked() {
                    state.request_reset_camera = true;
                }
            });

            ui.separator();
            ui.small("Controls:\n- RMB Drag: Orbit Camera\n- MMB Drag: Pan Camera\n- Scroll: Zoom\n- LMB Drag: Move Cage Vertex");
        });
}
