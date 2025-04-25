use crate::cpu_backend::CpuDevice;
use crate::{DType, Device, Result, WithDType};

//solution for fast cpu offloading
#[derive(Clone, Debug)]
pub struct OffloadBuffer {
    ptr_host: *mut core::ffi::c_void,
    len: usize,
    pub cpu_device: CpuDevice,
    pub from_device: crate::Device,
    dtype: DType,
}

unsafe impl Send for OffloadBuffer {}
unsafe impl Sync for OffloadBuffer {}

impl OffloadBuffer {
    pub fn new<T: WithDType>(
        src: &[T],
        dtype: DType,
        cpu_device: &CpuDevice,
        from_device: &Device,
    ) -> Result<Self> {
        use crate::gcu_backend::ubridge::gcu_device::driv;
        let size = std::mem::size_of::<T>() * src.len();
        let mut ptr_host = std::ptr::null_mut();
        unsafe {
            driv::topsHostMalloc(&mut ptr_host as *mut *mut core::ffi::c_void, size as u64, 0);
            std::ptr::copy(src.as_ptr() as *mut core::ffi::c_void, ptr_host, size);
        }
        Ok(OffloadBuffer {
            ptr_host,
            dtype,
            len: src.len(),
            cpu_device: cpu_device.clone(),
            from_device: from_device.clone(),
        })
    }

    pub fn reload(&self) -> Result<crate::Storage> {
        use crate::{Device, Storage};
        use half::{bf16, f16};
        let storage = match &self.from_device {
            Device::Gcu(dev) => match self.dtype {
                DType::BF16 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut bf16, self.len)?)
                }
                DType::F16 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut f16, self.len)?)
                }
                DType::F32 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut f32, self.len)?)
                }
                DType::U8 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut u8, self.len)?)
                }
                DType::I8 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut i8, self.len)?)
                }
                DType::U32 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut u32, self.len)?)
                }
                DType::I32 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut i32, self.len)?)
                }
                DType::I64 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut i64, self.len)?)
                }
                DType::F64 => {
                    Storage::Gcu(dev.storage_from_buffer(self.ptr_host as *mut f64, self.len)?)
                }
            },
            _ => {
                panic!("not supported device for cpu offloading")
            }
        };
        Ok(storage)
    }
}

impl Drop for OffloadBuffer {
    fn drop(&mut self) {
        use crate::gcu_backend::ubridge::gcu_device::driv;
        unsafe {
            if !self.ptr_host.is_null() {
                driv::topsHostFree(self.ptr_host);
            }
        }
    }
}
