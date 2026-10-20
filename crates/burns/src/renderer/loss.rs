//! A single device-loss observation shared by live renderers on the same device.

use std::sync::{Arc, Mutex, PoisonError, Weak};

use super::RenderError;

type LossState = Mutex<Option<String>>;

#[cfg(not(target_arch = "wasm32"))]
type DeviceKey = (wgpu::Backend, usize);
#[cfg(target_arch = "wasm32")]
type DeviceKey = wgpu::Device;

#[cfg(not(target_arch = "wasm32"))]
fn native_key<A: wgpu::hal::Api>(device: &wgpu::Device) -> Option<usize> {
    // SAFETY: only observe the HAL wrapper's address; never access or destroy its
    // raw handle. The registry retains the device while this address is a key.
    unsafe { device.as_hal::<A>() }.map(|guard| (&*guard as *const A::Device).addr())
}

#[cfg(not(target_arch = "wasm32"))]
fn device_key(device: &wgpu::Device) -> Result<DeviceKey, RenderError> {
    let backend = device.adapter_info().backend;
    // wgpu 30 native Device equality uses IDs local to an Instance. Independent
    // headless contexts can have equal IDs, so key the live HAL wrapper instead.
    let identity = match backend {
        #[cfg(target_vendor = "apple")]
        wgpu::Backend::Metal => native_key::<wgpu::hal::api::Metal>(device),
        #[cfg(not(target_vendor = "apple"))]
        wgpu::Backend::Vulkan => native_key::<wgpu::hal::api::Vulkan>(device),
        #[cfg(target_os = "windows")]
        wgpu::Backend::Dx12 => native_key::<wgpu::hal::api::Dx12>(device),
        _ => None,
    };
    identity
        .map(|address| (backend, address))
        .ok_or_else(|| RenderError::NoAdapter {
            backends: format!("{:?}", super::HEADLESS_BACKENDS),
            message: format!("the supplied device uses unsupported backend {backend:?}"),
        })
}

#[cfg(target_arch = "wasm32")]
fn device_key(device: &wgpu::Device) -> Result<DeviceKey, RenderError> {
    Ok(device.clone())
}

struct DeviceWatch {
    key: DeviceKey,
    #[cfg(not(target_arch = "wasm32"))]
    _device: wgpu::Device,
    lost: Weak<LossState>,
}

#[cfg(not(target_arch = "wasm32"))]
static WATCHES: Mutex<Vec<DeviceWatch>> = Mutex::new(Vec::new());

#[cfg(not(target_arch = "wasm32"))]
fn with_watches<R>(consume: impl FnOnce(&mut Vec<DeviceWatch>) -> R) -> R {
    consume(&mut WATCHES.lock().unwrap_or_else(PoisonError::into_inner))
}

// Browser GPU handles are local to their JS thread, not Send/Sync.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static WATCHES: std::cell::RefCell<Vec<DeviceWatch>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "wasm32")]
fn with_watches<R>(consume: impl FnOnce(&mut Vec<DeviceWatch>) -> R) -> R {
    WATCHES.with(|watches| consume(&mut watches.borrow_mut()))
}

pub(super) struct DeviceLoss {
    state: Option<Arc<LossState>>,
}

impl DeviceLoss {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self, RenderError> {
        let key = device_key(device)?;
        with_watches(|watches| {
            watches.retain(|watch| watch.lost.strong_count() != 0);
            if let Some(state) = watches
                .iter()
                .find(|watch| watch.key == key)
                .and_then(|watch| watch.lost.upgrade())
            {
                return Ok(Self { state: Some(state) });
            }
            let state = Arc::new(Mutex::new(None));
            let lost = Arc::downgrade(&state);
            device.set_device_lost_callback(move |reason, message| {
                if let Some(state) = lost.upgrade() {
                    *state.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(format!("{reason:?}: {message}"));
                }
            });
            watches.push(DeviceWatch {
                key,
                #[cfg(not(target_arch = "wasm32"))]
                _device: device.clone(),
                lost: Arc::downgrade(&state),
            });
            Ok(Self { state: Some(state) })
        })
    }

    pub(super) fn check(&self) -> Result<(), RenderError> {
        let state = self.state.as_ref().expect("device watch is live");
        match &*state.lock().unwrap_or_else(PoisonError::into_inner) {
            Some(reason) => Err(RenderError::DeviceLost {
                reason: reason.clone(),
            }),
            None => Ok(()),
        }
    }
}

impl Drop for DeviceLoss {
    fn drop(&mut self) {
        // Drop this strong reference before pruning: simultaneous renderer drops
        // cannot leave a dead registry entry retaining the device indefinitely.
        drop(self.state.take());
        with_watches(|watches| watches.retain(|watch| watch.lost.strong_count() != 0));
    }
}
