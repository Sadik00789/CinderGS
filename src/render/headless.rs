//! Offscreen rendering used by the CLI render farm and the demo recorder.

use crate::deformation::CageScene;
use crate::render::blit::OFFSCREEN_FORMAT;
use crate::render::camera::CameraUniforms;
use crate::render::pipeline::{ExecutionBackend, FrameResult, RenderPipelineOrchestrator};
use crate::render::rasterizer::CompositorUniforms;
use crate::scene::GaussianSceneSoa;

/// A headless WGPU device plus the shared render orchestrator.
pub struct HeadlessRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub orchestrator: RenderPipelineOrchestrator,
    width: u32,
    height: u32,
}

impl HeadlessRenderer {
    /// Creates a headless device, preferring a high-performance adapter and
    /// falling back to a low-power or software adapter before erroring out.
    pub fn new(width: u32, height: u32, backend: ExecutionBackend) -> anyhow::Result<Self> {
        let width = width.max(1);
        let height = height.max(1);

        let instance = wgpu::Instance::default();
        let adapter = Self::request_adapter(&instance)?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("CinderGSHeadlessDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|err| anyhow::anyhow!("failed to create WGPU device on {}: {err}", info.name))?;

        let mut orchestrator = RenderPipelineOrchestrator::new(&device, OFFSCREEN_FORMAT);
        orchestrator.resize(&device, width, height);
        orchestrator.backend = backend;

        match orchestrator.attach_shared_gpu(
            instance,
            adapter,
            device.clone(),
            queue.clone(),
            info.backend,
        ) {
            Ok(()) => {}
            Err(err) => {
                eprintln!(
                    "warning: could not share the presentation device with CubeCL ({err}); \
                     the GPU backend will acquire its own device on demand"
                );
            }
        }

        Ok(Self {
            device,
            queue,
            orchestrator,
            width,
            height,
        })
    }

    fn request_adapter(instance: &wgpu::Instance) -> anyhow::Result<wgpu::Adapter> {
        let high = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));

        if let Ok(adapter) = high {
            return Ok(adapter);
        }

        let low = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));

        if let Ok(adapter) = low {
            return Ok(adapter);
        }

        let fallback = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: true,
            apply_limit_buckets: false,
        }));

        fallback.map_err(|err| anyhow::anyhow!("no usable WGPU adapter: {err}"))
    }

    /// Output resolution.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resizes the offscreen target.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.orchestrator
            .resize(&self.device, self.width, self.height);
    }

    /// Human-readable compute device, when available.
    pub fn gpu_description(&self) -> Option<String> {
        self.orchestrator.compute_description()
    }

    /// Renders one frame entirely offscreen and returns the composited image.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        show_wireframe: bool,
        selected: Option<(usize, usize)>,
    ) -> anyhow::Result<FrameResult> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("HeadlessFrameEncoder"),
            });

        let result = self.orchestrator.render_frame(
            &self.device,
            &self.queue,
            &mut encoder,
            scene,
            cages,
            camera,
            compositor,
            None,
            show_wireframe,
            selected,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| anyhow::anyhow!("WGPU poll failed: {err}"))?;

        Ok(result)
    }
}
