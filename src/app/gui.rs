use crate::app::timeline::Timeline;
use crate::deformation::CageScene;
use crate::render::ExecutionBackend;

/// Immediate-mode UI state and performance telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct GuiState {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub splat_count: usize,
    pub culled_count: usize,
    pub active_tiles: usize,
    pub sort_reused: bool,
    pub selected: Option<(usize, usize)>,
    pub sh_degree: u32,
    pub show_cage: bool,
    pub vsync: bool,
    pub backend: ExecutionBackend,
    pub active_cage: usize,
    pub trigger_jiggle: bool,
    pub request_reset_camera: bool,
    pub request_reset_cage: bool,
    pub add_cage: bool,
    pub remove_cage: bool,
    pub bake_keyframe: bool,
    pub clear_timeline: bool,
    pub toggle_pin_selected: bool,
    pub clear_pins: bool,
    pub ground_collision_enabled: bool,
    pub ground_collision_y: f32,
    pub strain_split_threshold: f32,
    pub mip_filter_3d_var: f32,
    pub export_deformed_ply: bool,
    pub export_timeline_renders: bool,
    pub status_message: Option<String>,
    pub gpu_description: Option<String>,
    pub gpu_error: Option<String>,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            fps: 60.0,
            frame_time_ms: 16.6,
            splat_count: 0,
            culled_count: 0,
            active_tiles: 0,
            sort_reused: false,
            selected: None,
            sh_degree: 3,
            show_cage: true,
            vsync: true,
            backend: ExecutionBackend::CpuReference,
            active_cage: 0,
            trigger_jiggle: false,
            request_reset_camera: false,
            request_reset_cage: false,
            add_cage: false,
            remove_cage: false,
            bake_keyframe: false,
            clear_timeline: false,
            toggle_pin_selected: false,
            clear_pins: false,
            ground_collision_enabled: false,
            ground_collision_y: -1.0,
            strain_split_threshold: 0.0,
            mip_filter_3d_var: 0.0,
            export_deformed_ply: false,
            export_timeline_renders: false,
            status_message: None,
            gpu_description: None,
            gpu_error: None,
        }
    }
}

/// Renders the immediate-mode HUD overlay using egui.
pub fn render_gui(
    ctx: &egui::Context,
    state: &mut GuiState,
    cages: &mut CageScene,
    timeline: &mut Timeline,
) {
    egui::Window::new("CinderGS Engine Telemetry")
        .default_pos(egui::pos2(16.0, 16.0))
        .default_size(egui::vec2(320.0, 520.0))
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
            ui.label(format!(
                "Stage 3 Sort: {}",
                if state.sort_reused {
                    "reused (camera/cage static)"
                } else {
                    "recomputed"
                }
            ));

            if let Some(description) = &state.gpu_description {
                ui.small(format!("Compute: {description}"));
            }
            if let Some(error) = &state.gpu_error {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 140, 120),
                    format!("GPU unavailable: {error}"),
                );
            }

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
            ui.horizontal(|ui| {
                ui.label("Split Threshold:");
                ui.add(
                    egui::Slider::new(&mut state.strain_split_threshold, 0.0..=3.0)
                        .fixed_decimals(2),
                );
            });
            ui.horizontal(|ui| {
                ui.label("3D Mip Var:");
                ui.add(
                    egui::Slider::new(&mut state.mip_filter_3d_var, 0.0..=0.01)
                        .fixed_decimals(4),
                );
            });
            ui.checkbox(&mut state.show_cage, "Show Cage Wireframe");
            ui.checkbox(&mut state.vsync, "VSync (Uncap FPS when off)");

            ui.separator();
            ui.heading("Volumetric Cages");

            ui.horizontal(|ui| {
                for (index, cage) in cages.cages.iter().enumerate() {
                    ui.selectable_value(
                        &mut state.active_cage,
                        index,
                        format!("{} {}", index, cage.name),
                    );
                }
            });

            if state.active_cage >= cages.cages.len() {
                state.active_cage = 0;
            }

            if let Some(cage) = cages.cages.get_mut(state.active_cage) {
                ui.horizontal(|ui| {
                    ui.label("Stiffness:");
                    ui.add(
                        egui::Slider::new(&mut cage.simulator.stiffness, 10.0..=500.0)
                            .text("N/m"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Damping:");
                    ui.add(egui::Slider::new(&mut cage.simulator.damping, 0.50..=0.99));
                });

                ui.horizontal(|ui| {
                    if ui.button("Jiggle Impulse").clicked() {
                        state.trigger_jiggle = true;
                    }
                    if ui.button("Reset Cage").clicked() {
                        state.request_reset_cage = true;
                    }
                });
                ui.small(format!("Pinned handles: {}", cage.simulator.pinned_count()));
            }

            ui.horizontal(|ui| {
                ui.checkbox(&mut state.ground_collision_enabled, "Ground Plane Y:");
                if state.ground_collision_enabled {
                    ui.add(
                        egui::DragValue::new(&mut state.ground_collision_y)
                            .speed(0.05)
                            .fixed_decimals(2),
                    );
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Add Cage").clicked() {
                    state.add_cage = true;
                }
                if ui.button("Remove Cage").clicked() {
                    state.remove_cage = true;
                }
            });

            ui.separator();
            ui.heading("Keyframe Timeline & Export");

            ui.horizontal(|ui| {
                ui.checkbox(&mut timeline.playing, "Play");
                let duration = timeline.duration().max(0.001);
                ui.add(
                    egui::Slider::new(&mut timeline.playhead, 0.0..=duration)
                        .text("s")
                        .fixed_decimals(2),
                );
            });
            ui.horizontal(|ui| {
                if ui.button("Bake Keyframe").clicked() {
                    state.bake_keyframe = true;
                }
                if ui.button("Clear Timeline").clicked() {
                    state.clear_timeline = true;
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Export Deformed PLY").clicked() {
                    state.export_deformed_ply = true;
                }
                if ui.button("Export Renders").clicked() {
                    state.export_timeline_renders = true;
                }
            });
            ui.small(format!(
                "{} keyframe(s) — scrubbing applies the interpolated camera and cage pose",
                timeline.keyframes.len()
            ));
            if let Some(status) = &state.status_message {
                ui.small(status);
            }

            ui.separator();
            ui.heading("Cage Interaction");
            if let Some((cage, vertex)) = state.selected {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 255, 100),
                    format!("Selected: cage {cage}, vertex {vertex}"),
                );
                ui.horizontal(|ui| {
                    if ui.button("Pin / Unpin Handle").clicked() {
                        state.toggle_pin_selected = true;
                    }
                    if ui.button("Clear Pins").clicked() {
                        state.clear_pins = true;
                    }
                });
                ui.label("Drag with Left Mouse Button to deform");
            } else {
                ui.label("Click a cage handle to pick and deform");
                if ui.button("Clear Pins").clicked() {
                    state.clear_pins = true;
                }
            }
            if ui.button("Reset Camera").clicked() {
                state.request_reset_camera = true;
            }

            ui.separator();
            ui.small("Controls:\n- RMB Drag: Orbit Camera\n- MMB Drag: Pan Camera\n- Scroll: Zoom\n- LMB Drag: Move Cage Vertex");
        });
}
