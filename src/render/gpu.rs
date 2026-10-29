//! One-time CubeCL/WGPU device acquisition.
//!
//! The engine keeps a single compute client for the whole process. It is created
//! lazily on the first GPU frame (or explicitly through [`GpuCompute::from_shared`]
//! so the compute kernels can share the presentation `wgpu::Device`), and it is
//! never rebuilt per frame and never guarded by `catch_unwind`.

use cubecl::prelude::Client;
use cubecl::wgpu::RuntimeOptions;
use thiserror::Error;

/// Failure to bring up the CubeCL WGPU runtime.
#[derive(Debug, Error)]
pub enum GpuError {
    /// The CubeCL WGPU runtime refused to initialise.
    #[error("CubeCL WGPU runtime initialisation failed: {0}")]
    Runtime(#[from] cubecl::wgpu::WgpuInitError),

    /// No adapter of the requested class exists on this machine.
    #[error("no usable GPU adapter found: {0}")]
    NoAdapter(String),

    /// The runtime is not compiled for this target (e.g. `wasm32`).
    #[error("GPU compute is unavailable on this target")]
    UnsupportedTarget,
}

/// A persistent CubeCL client plus human-readable device information.
pub struct GpuCompute {
    client: Client,
    adapter_name: String,
    backend_name: String,
}

impl std::fmt::Debug for GpuCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCompute")
            .field("adapter", &self.adapter_name)
            .field("backend", &self.backend_name)
            .finish_non_exhaustive()
    }
}

impl GpuCompute {
    /// Acquires the best available adapter exactly once.
    ///
    /// Preference order is discrete GPU, then integrated GPU, then whatever the
    /// runtime considers the default device. Every step is a fallible call, so a
    /// machine without a usable adapter produces an error instead of a panic.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn acquire() -> Result<Self, GpuError> {
        use cubecl::wgpu::{AutoGraphicsApi, WgpuDevice, WgpuDeviceKind};

        let mut last_error = String::from("no adapter classes were reported");

        for kind in [
            WgpuDeviceKind::DiscreteGpu(0),
            WgpuDeviceKind::IntegratedGpu(0),
            WgpuDeviceKind::DefaultDevice,
        ] {
            let device = WgpuDevice::new(kind);
            let options = RuntimeOptions::try_default()?;
            match cubecl::wgpu::try_init_setup::<AutoGraphicsApi>(&device, options) {
                Ok(setup) => {
                    let info = setup.adapter.get_info();
                    let client = cubecl::Device::Wgpu(device).client();
                    return Ok(Self {
                        client,
                        adapter_name: info.name,
                        backend_name: format!("{:?}", info.backend),
                    });
                }
                Err(err) => last_error = err.to_string(),
            }
        }

        Err(GpuError::NoAdapter(last_error))
    }

    /// Acquires a CubeCL client on a target that cannot provide one.
    #[cfg(target_arch = "wasm32")]
    pub fn acquire() -> Result<Self, GpuError> {
        Err(GpuError::UnsupportedTarget)
    }

    /// Registers an already-created `wgpu` device with CubeCL.
    ///
    /// Sharing one `wgpu::Device`/`Queue` between the compute kernels and the
    /// presentation surface removes the duplicate WGPU stack and lets compute
    /// output live in the same GPU memory as the blit source.
    pub fn from_shared(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        backend: wgpu::Backend,
    ) -> Result<Self, GpuError> {
        let info = adapter.get_info();
        let setup = cubecl::wgpu::WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend,
        };
        let options = RuntimeOptions::try_default()?;
        let cube_device = cubecl::wgpu::try_init_device(setup, options)?;
        let client = cubecl::Device::Wgpu(cube_device).client();

        Ok(Self {
            client,
            adapter_name: info.name,
            backend_name: format!("{:?}", info.backend),
        })
    }

    /// The persistent CubeCL client.
    #[inline]
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Adapter name reported by WGPU.
    #[inline]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Graphics backend name (Vulkan, Metal, Dx12, ...).
    #[inline]
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Human-readable description used in telemetry and benchmark output.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.adapter_name, self.backend_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_is_fallible_and_stable() {
        // Either a GPU exists (and the client is reusable) or the call reports an
        // error; it must never panic.
        match GpuCompute::acquire() {
            Ok(gpu) => {
                assert!(!gpu.adapter_name().is_empty());
                let _ = gpu.describe();
                let _ = gpu.client();
            }
            Err(err) => {
                let message = err.to_string();
                assert!(!message.is_empty());
            }
        }
    }
}
