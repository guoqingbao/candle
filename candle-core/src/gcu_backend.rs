/*
* Copyright 2021-2024 Enflame. All Rights Reserved.

* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* @file    device_executor.rs
* @brief
*
* @author  Guoqing Bao
* @date    2022-09-05 - 2024-02-02
* @version V0.1
* @par     Copyright (c) Enflame Tech Company.
* @par     History: Support BigCode/StarCode model inference
* @par     Comments: a gcu backend for huggingface candle ML framework,
*                    aiming at minimal modification of upstream candle project while supporting
*                    Enflame GCU device computing seamlessly. This GCU backend requires another two
*                    crates: ubridge and UHHI (written by Enflame), which are not open source yet.
*/
use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuStorageRef;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
pub use cust_core::_hidden::DeviceCopy;
use half::{bf16, f16};
use std::sync::Arc;
pub use ubridge;
use ubridge::gcu_device::GcuDevice as RawDevice;
use ubridge::gcu_launch::GcuLaunchAsync;
use ubridge::gcu_slice::{GcuSlice, GcuView, GcuViewMut};
use ubridge::prelude::{DevicePtr, DeviceSlice};
use uhal::error::DeviceError;
use uhal::memory::DevicePointerTrait;

#[derive(Debug, Clone)]
pub enum GcuStorageSlice {
    U8(GcuSlice<u8>),
    I8(GcuSlice<i8>),
    U32(GcuSlice<u32>),
    I32(GcuSlice<i32>),
    I64(GcuSlice<i64>),
    BF16(GcuSlice<bf16>),
    F16(GcuSlice<f16>),
    F32(GcuSlice<f32>),
    F64(GcuSlice<f64>),
}

type S = GcuStorageSlice;

#[derive(Debug, Clone)]
pub struct GcuStorage {
    pub slice: GcuStorageSlice,
    pub device: GcuDevice,
}

#[derive(Debug, Clone)]
pub struct GcuDevice {
    device: Arc<RawDevice>,
}

/// Gcurc related errors
#[derive(thiserror::Error, Debug)]
pub enum GcuError {
    // #[error(transparent)]
    // Gcu(#[from] Gcurc::driver::DriverError),

    // #[error(transparent)]
    // Compiler(#[from] Gcurc::nvrtc::CompileError),
    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{gcu} when loading {module_name}")]
    Load { gcu: String, module_name: String },
}

impl From<GcuError> for crate::Error {
    fn from(val: GcuError) -> Self {
        crate::Error::Gcu(Box::new(val)).bt()
    }
}

impl From<DeviceError> for crate::Error {
    fn from(val: DeviceError) -> Self {
        crate::Error::Gcu(Box::new(val)).bt()
    }
}

// /// Unique identifier for Gcu devices.
// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// pub struct DeviceId(usize);

// impl DeviceId {
//     fn new() -> Self {
//         // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
//         use std::sync::atomic;
//         static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
//         Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
//     }
// }

// struct GcuRng(Gcurc::curand::GcuRng);
// unsafe impl Send for GcuRng {}

impl std::ops::Deref for GcuDevice {
    type Target = Arc<RawDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl std::ops::DerefMut for GcuDevice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.device
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

// impl<O, E: Into<GcuError>> WrapErr<O> for std::result::Result<O, E> {
//     fn w(self) -> std::result::Result<O, crate::Error> {
//         self.map_err(|e| crate::Error::Gcu(Box::new(e.into())))
//     }
// }

impl<O, E: Into<DeviceError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Gcu(Box::new(e.into())))
    }
}

impl GcuDevice {
    pub fn gcu_device(&self) -> Arc<RawDevice> {
        self.device.clone()
    }

    pub fn id(&self) -> usize {
        self.device.id
    }

    fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<GcuStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<u8>(elem_count).w()?;
                let func = self.get_or_load_func("fill_u8", ubridge::FILL)?;
                let params = (data.device_ptr(), v as u8, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::U8(data)
            }
            DType::I8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<i8>(elem_count).w()?;
                let func = self.get_or_load_func("fill_i8", ubridge::FILL)?;
                let params = (data.device_ptr(), v as u8, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::I8(data)
            }
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<u32>(elem_count).w()?;
                let func = self.get_or_load_func("fill_u32", ubridge::FILL)?;
                let params = (data.device_ptr(), v as u32, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::U32(data)
            }
            DType::I32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<i32>(elem_count).w()?;
                let func = self.get_or_load_func("fill_i32", ubridge::FILL)?;
                let params = (data.device_ptr(), v as i32, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::I32(data)
            }
            DType::I64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<i64>(elem_count).w()?;
                let func = self.get_or_load_func("fill_i64", ubridge::FILL)?;
                let params = (data.device_ptr(), v as i64, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::I64(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<bf16>(elem_count).w()?;
                let func = self.get_or_load_func("fill_bf16", ubridge::FILL)?;
                let params = (data.device_ptr(), bf16::from_f64(v), elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f16>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f16", ubridge::FILL)?;
                let params = (data.device_ptr(), f16::from_f64(v), elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f32>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f32", ubridge::FILL)?;
                let params = (data.device_ptr(), v as f32, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f64>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f64", ubridge::FILL)?;
                let params = (data.device_ptr(), v, elem_count);
                unsafe { func.launch(&self.launch_cfg, params).w()? };
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    // pub fn get_or_load_func(&self, module_name: &str, kernel_path: &str) -> Result<GcuFunction> {
    // Ok(GcuFunction::new(module_name.to_string(), kernel_path.to_string()))
    //     if !self.has_func(module_name) {
    //         // Leaking the string here is a bit sad but we need a &'static str and this is only
    //         // done once per kernel name.
    //         let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
    //         self.load_kernel(kernel_path, module_name, &[static_module_name])
    //             .map_err(|Gcu| GcuError::Load {
    //                 Gcu,
    //                 module_name: module_name.to_string(),
    //             })
    //             ?;
    //     }
    //     self.get_func(module_name)
    //         // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
    //         // able to only build the error value if needed.
    //         .ok_or(GcuError::MissingKernel {
    //             module_name: module_name.to_string(),
    //         })

    // }

    pub fn storage_from_buffer<T: crate::WithDType>(
        &self,
        s: *mut T,
        len: usize,
    ) -> Result<GcuStorage> {
        let ss = unsafe { std::slice::from_raw_parts(s, len) };
        let slice = match T::cpu_storage_ref(ss) {
            CpuStorageRef::U8(_) => {
                let data = self.htod_copy_buffer::<u8>(s as *mut u8, len).w()?;
                GcuStorageSlice::U8(data)
            }
            CpuStorageRef::I8(_) => {
                let data = self.htod_copy_buffer::<i8>(s as *mut i8, len).w()?;
                GcuStorageSlice::I8(data)
            }
            CpuStorageRef::U32(_) => {
                let data = self.htod_copy_buffer::<u32>(s as *mut u32, len).w()?;
                GcuStorageSlice::U32(data)
            }
            CpuStorageRef::I32(_) => {
                let data = self.htod_copy_buffer::<i32>(s as *mut i32, len).w()?;
                GcuStorageSlice::I32(data)
            }
            CpuStorageRef::I64(_) => {
                let data = self.htod_copy_buffer::<i64>(s as *mut i64, len).w()?;
                GcuStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(_) => {
                let data = self.htod_copy_buffer::<bf16>(s as *mut bf16, len).w()?;
                GcuStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(_) => {
                let data = self.htod_copy_buffer::<f16>(s as *mut f16, len).w()?;
                GcuStorageSlice::F16(data)
            }
            CpuStorageRef::F32(_) => {
                let data = self.htod_copy_buffer::<f32>(s as *mut f32, len).w()?;
                GcuStorageSlice::F32(data)
            }
            CpuStorageRef::F64(_) => {
                let data = self.htod_copy_buffer::<f64>(s as *mut f64, len).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }
}

impl BackendDevice for GcuDevice {
    type Storage = GcuStorage;
    fn new(ordinal: usize) -> Result<Self> {
        Ok(GcuDevice {
            #[cfg(feature = "async")]
            device: RawDevice::new(ordinal, false).unwrap(),
            #[cfg(not(feature = "async"))]
            device: RawDevice::new(ordinal, true).unwrap(),
        })
    }
    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Gcu {
            gpu_id: self.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<GcuStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count).w()?;
                GcuStorageSlice::U8(data)
            }
            DType::I8 => {
                let data = self.alloc_zeros::<i8>(elem_count).w()?;
                GcuStorageSlice::I8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count).w()?;
                GcuStorageSlice::U32(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count).w()?;
                GcuStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count).w()?;
                GcuStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count).w()?;
                GcuStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count).w()?;
                GcuStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count).w()?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<GcuStorage> {
        let elem_count = shape.elem_count();
        // let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // Gcurc changes.
            DType::U8
            | DType::I8
            | DType::U32
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(GcuError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })?,
            DType::F32 => {
                let data = self.device.alloc::<f32>(elem_count).w()?;
                // curand.0.fill_with_uniform(&mut data)?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.device.alloc::<f64>(elem_count).w()?;
                // curand.0.fill_with_uniform(&mut data)?;
                GcuStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            let layout = Layout::contiguous(shape);
            Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<GcuStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // Gcurc changes.
        let elem_count = shape.elem_count();
        // let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            DType::U8
            | DType::I8
            | DType::U32
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(GcuError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })?,
            DType::F32 => {
                let data = self.device.alloc::<f32>(elem_count).w()?;
                // curand
                //     .0
                //     .fill_with_normal(&mut data, mean as f32, std as f32)
                //     ?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.device.alloc::<f64>(elem_count).w()?;
                // curand.0.fill_with_normal(&mut data, mean, std)?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<GcuStorage> {
        self.const_impl(1., shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                GcuStorageSlice::U8(data)
            }
            DType::I8 => {
                let data = self.alloc::<i8>(elem_count).w()?;
                GcuStorageSlice::I8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count).w()?;
                GcuStorageSlice::U32(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count).w()?;
                GcuStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count).w()?;
                GcuStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count).w()?;
                GcuStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count).w()?;
                GcuStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count).w()?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U8(data)
            }
            CpuStorageRef::I8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U32(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<GcuStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U8(data)
            }
            CpuStorage::I8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U32(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<GcuStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::U8(data)
            }
            CpuStorage::I8(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::I8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::U32(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_copy(storage).w()?;
                GcuStorageSlice::F64(data)
            }
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}

pub trait Map1 {
    fn f<T: DeviceCopy + WithDType>(
        //+ValidAsZeroBits
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>>;

    fn map(&self, s: &S, d: &GcuDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::I8(s) => S::I8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I32(s) => S::I32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
        };
        Ok(out)
    }
}

trait Map2 {
    fn f<T: DeviceCopy + WithDType>(
        //+ValidAsZeroBits
        &self,
        src1: &GcuSlice<T>,
        layout1: &Layout,
        src2: &GcuSlice<T>,
        layout2: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &GcuDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => S::U8(self.f(s1, l1, s2, l2, d)?),
            (S::U32(s1), S::U32(s2)) => S::U32(self.f(s1, l1, s2, l2, d)?),
            (S::I32(s1), S::I32(s2)) => S::I32(self.f(s1, l1, s2, l2, d)?),
            (S::I64(s1), S::I64(s2)) => S::I64(self.f(s1, l1, s2, l2, d)?),
            (S::BF16(s1), S::BF16(s2)) => S::BF16(self.f(s1, l1, s2, l2, d)?),
            (S::F16(s1), S::F16(s2)) => S::F16(self.f(s1, l1, s2, l2, d)?),
            (S::F32(s1), S::F32(s2)) => S::F32(self.f(s1, l1, s2, l2, d)?),
            (S::F64(s1), S::F64(s2)) => S::F64(self.f(s1, l1, s2, l2, d)?),
            _ => Err(GcuError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}

trait Map2InPlace {
    fn f<T: DeviceCopy + WithDType>(
        //+ValidAsZeroBits
        &self,
        dst: &mut GcuSlice<T>,
        dst_shape: &Shape,
        src: &GcuSlice<T>,
        src_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<()>;

    fn map(
        &self,
        dst: &mut S,
        dst_s: &Shape,
        src: &S,
        src_l: &Layout,
        d: &GcuDevice,
    ) -> Result<()> {
        match (dst, src) {
            (S::U8(dst), S::U8(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::U32(dst), S::U32(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::I32(dst), S::I32(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_s, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_s, src, src_l, d),
            _ => Err(GcuError::InternalError("dtype mismatch in binary op"))?,
        }
    }
}

trait Map1Any {
    fn f<T: DeviceCopy + WithDType, W: Fn(GcuSlice<T>) -> S>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &GcuDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::I8(s) => self.f(s, d, l, S::I8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I32(s) => self.f(s, d, l, S::I32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
        };
        Ok(out)
    }
}

trait Map2Any {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src1: &GcuSlice<T>,
        layout1: &Layout,
        src2: &GcuSlice<T>,
        layout2: &Layout,
        dev: &GcuDevice,
    ) -> Result<S>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &GcuDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::U32(s1), S::U32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::I32(s1), S::I32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::I64(s1), S::I64(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::BF16(s1), S::BF16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F16(s1), S::F16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F32(s1), S::F32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F64(s1), S::F64(s2)) => self.f(s1, l1, s2, l2, d)?,
            _ => Err(GcuError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}

struct Clone;
impl Map1 for Clone {
    fn f<T: DeviceCopy>(&self, s: &GcuSlice<T>, _: &GcuDevice, _: &Layout) -> Result<GcuSlice<T>> {
        s.try_clone().w()
    }
}

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

struct Affine(f64, f64);
impl Map1 for Affine {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let el = shape.elem_count();
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("affine"), ubridge::AFFINE)?;
        let out = dev.alloc::<T>(el)?;
        let params = (
            el,
            src.device_ptr(),
            out.device_ptr(),
            self.0 as f32,
            self.1 as f32,
        );
        unsafe { func.launch(&dev.launch_cfg, params) }?;
        Ok(out)
    }
}

struct Elu(f64);
impl Map1 for Elu {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("uelu"), ubridge::UNARY)?;
        let out = dev.alloc::<T>(el)?;
        let mut cfg = dev.launch_cfg.clone();
        cfg.set_shared_memory(src.num_bytes() as u32);
        let params = (
            el,
            dims.len(),
            ds.device_ptr(),
            T::from_f64(self.0),
            src.device_ptr(),
            out.device_ptr(),
        );
        unsafe { func.launch(&cfg, params) }?;
        Ok(out)
    }
}

struct Powf(f64);
impl Map1 for Powf {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = &dev.launch_cfg;
        let ds = dev.htod_copy([dims, layout.stride()].concat()).w()?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("upowf"), ubridge::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el).w()?;
        let params = (el, dims.len(), &ds, T::from_f64(self.0), src, &out);
        // SAFETY: ffi.
        unsafe { func.launch(&cfg, params) }.w()?;
        Ok(out)
    }
}

struct Sum<'a>(&'a [usize]);
impl<'a> Map1 for Sum<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let src_dims = shape.dims();
        let el = shape.elem_count();
        let mut dst_el = el;
        for &sum_dim in self.0.iter() {
            dst_el /= src_dims[sum_dim];
        }
        let mut sum_dims = self.0.to_vec();
        // Sort the sum_dims as they have to be processed from left to right when converting the
        // indexes.
        sum_dims.sort();
        let sum_dims_l: Vec<usize> = sum_dims.iter().map(|&d| src_dims[d]).collect();
        let sum_dims_s: Vec<usize> = sum_dims
            .iter()
            .map(|&d| src_dims[d + 1..].iter().product::<usize>())
            .collect();
        let ds = dev.htod_copy([src_dims, layout.stride(), &sum_dims_l, &sum_dims_s].concat())?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("sum"), ubridge::REDUCE)?;
        let out = dev.alloc_zeros::<T>(dst_el).w()?;
        let params = (
            el,
            src_dims.len(),
            sum_dims.len(),
            ds.device_ptr(),
            src.device_ptr(),
            out.device_ptr(),
        );
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct FastReduce<'a>(&'a [usize], ReduceOp);
impl<'a> Map1Any for FastReduce<'a> {
    fn f<T: DeviceCopy + WithDType, W: Fn(GcuSlice<T>) -> S>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S> {
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        let src: &GcuView<'_, T> = &src.slice(layout.start_offset()..);
        let (name, check_empty, return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false, false),
            ReduceOp::Min => ("fast_min", true, false),
            ReduceOp::Max => ("fast_max", true, false),
            ReduceOp::ArgMin => ("fast_argmin", true, true),
            ReduceOp::ArgMax => ("fast_argmax", true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let mut cfg = dev.launch_cfg.clone();
        cfg.set_shared_memory(src.num_bytes() as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::REDUCE)?;
        if return_index {
            let out = dev.alloc::<u32>(dst_el).w()?;
            let params = (
                src.device_ptr(),
                out.device_ptr(),
                src_el,
                el_to_sum_per_block,
            );
            unsafe { func.launch(&cfg, params) }.w()?;
            Ok(S::U32(out))
        } else {
            let out = dev.alloc::<T>(dst_el).w()?;
            let params = (
                src.device_ptr(),
                out.device_ptr(),
                src_el,
                el_to_sum_per_block,
            );
            unsafe { func.launch(&cfg, params) }.w()?;
            Ok(wrap(out))
        }
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>> {
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::UNARY)?;
        let out = dev.alloc::<T>(el_count).w()?;
        let params = (el_count, src.device_ptr(), out.device_ptr());
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct IndexSelect<'a>(&'a GcuStorage, &'a Layout, usize);
impl<'a> Map1 for IndexSelect<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        src_l: &Layout,
    ) -> Result<GcuSlice<T>> {
        let ids_l = &self.1;
        let ids_shape = ids_l.shape();
        let ids_el = ids_shape.elem_count();
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let dim_size = src_l.dims()[self.2];
        let out = dev.alloc::<T>(ids_el * left_size * right_size).w()?;

        match &self.0.slice {
            GcuStorageSlice::U32(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("is_u32"), ubridge::INDEXING)?;
                let params = (
                    ids_el,
                    ptr.device_ptr(),
                    src.device_ptr(),
                    out.device_ptr(),
                    left_size,
                    dim_size,
                    right_size,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            GcuStorageSlice::U8(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("is_u8"), ubridge::INDEXING)?;
                let params = (
                    ids_el,
                    ptr.device_ptr(),
                    src.device_ptr(),
                    out.device_ptr(),
                    left_size,
                    dim_size,
                    right_size,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            GcuStorageSlice::I64(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("is_i64"), ubridge::INDEXING)?;
                let params = (
                    ids_el,
                    ptr.device_ptr(),
                    src.device_ptr(),
                    out.device_ptr(),
                    left_size,
                    dim_size,
                    right_size,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "index_select ids should be u8 or u32",
                expected: DType::U32,
                got: self.0.dtype(),
            })?,
        };
        Ok(out)
    }
}

struct Gather<'a>(&'a GcuStorage, &'a Layout, usize);
impl<'a> Map1 for Gather<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        src_l: &Layout,
    ) -> Result<GcuSlice<T>> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            GcuStorageSlice::U32(slice) => ("gather_u32", slice.slice(ids_o1..ids_o2).device_ptr()),
            GcuStorageSlice::U8(slice) => ("gather_u8", slice.slice(ids_o1..ids_o2).device_ptr()),
            GcuStorageSlice::I64(slice) => ("gather_i64", slice.slice(ids_o1..ids_o2).device_ptr()),
            _ => Err(GcuError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let el = ids_l.shape().elem_count();
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        let out = dev.alloc::<T>(el).w()?;
        let params = (
            el,
            ids,
            src.device_ptr(),
            out.device_ptr(),
            left_sz,
            src_dim_sz,
            ids_dim_sz,
            right_sz,
        );
        let mut cfg = dev.launch_cfg.clone();
        cfg.set_shared_memory(el as u32 * std::mem::size_of::<T>() as u32);
        unsafe { func.launch(&cfg, params) }.w()?;
        Ok(out)
    }
}

struct IndexAdd<'a>(&'a GcuStorage, &'a Layout, usize);
impl<'a> Map2InPlace for IndexAdd<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        dst: &mut GcuSlice<T>,
        dst_shape: &Shape,
        src: &GcuSlice<T>,
        src_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let (name, ids) = match &ids.slice {
            GcuStorageSlice::U32(slice) => ("ia_u32", slice.slice(ids_o1..ids_o2).device_ptr()),
            GcuStorageSlice::I64(slice) => ("ia_i64", slice.slice(ids_o1..ids_o2).device_ptr()),
            GcuStorageSlice::U8(slice) => ("ia_u8", slice.slice(ids_o1..ids_o2).device_ptr()),
            _ => Err(GcuError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_shape.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];
        let el = (left_sz * right_sz) as u32;
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        let params = (
            ids,
            ids_dim_sz,
            src.device_ptr(),
            dst.device_ptr(),
            left_sz,
            src_dim_sz,
            dst_dim_sz,
            right_sz,
        );
        let mut cfg = dev.launch_cfg.clone();
        cfg.set_shared_memory(2 * el as u32 * std::mem::size_of::<T>() as u32);
        unsafe { func.launch(&cfg, params) }.w()?;
        Ok(())
    }
}

struct ScatterAdd<'a>(&'a GcuStorage, &'a Layout, usize);
impl<'a> Map2InPlace for ScatterAdd<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        dst: &mut GcuSlice<T>,
        dst_shape: &Shape,
        src: &GcuSlice<T>,
        src_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<()> {
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        // let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
        //     Some(o12) => o12,
        //     None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        // };
        let name = match &ids.slice {
            GcuStorageSlice::U32(_) => {
                // ("sa_u32", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "sa_u32"
            }
            GcuStorageSlice::I64(_) => {
                // ("sa_i64", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "sa_i64"
            }
            GcuStorageSlice::U8(_) => {
                // ("sa_u8", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "sa_u8"
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_shape.dims()[dim];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let params = (
            // ids,
            src.device_ptr(),
            dst.buffer,
            left_sz,
            src_dim_sz,
            dst_dim_sz,
            right_sz,
        );
        // SAFETY: ffi.
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(())
    }
}

struct Conv1D<'a>(&'a crate::conv::ParamsConv1D);
impl<'a> Map2 for Conv1D<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        inp: &GcuSlice<T>,
        inp_l: &Layout,
        k: &GcuSlice<T>,
        k_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>> {
        // Kernel shape: (c_out, c_in_k, k_size)
        // Input shape: (b_size, c_in, l_in) or (c_in, l_in)
        let p = &self.0;
        // let inp = &inp.slice(inp_l.start_offset()..);
        let src = &inp.slice(inp_l.start_offset()..);

        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let func = dev.get_or_load_func(&kernel_name::<T>("conv1d"), ubridge::CONV)?;
        let out = dev.alloc::<T>(dst_el).w()?;
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.htod_copy(ds).w()?;
        let params = (
            el,
            l_out,
            p.stride,
            p.padding,
            p.dilation,
            ds.device_ptr(),
            src.device_ptr(),
            k.device_ptr(),
            out.device_ptr(),
        );
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct Conv2D<'a>(&'a crate::conv::ParamsConv2D);
impl<'a> Map2 for Conv2D<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        inp: &GcuSlice<T>,
        inp_l: &Layout,
        k: &GcuSlice<T>,
        k_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>> {
        // Kernel shape: (c_out, c_in_k, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let func = dev.get_or_load_func(&kernel_name::<T>("conv2d"), ubridge::CONV)?;
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.htod_copy(ds).w()?;
        let params = (
            el,
            out_w,
            out_h,
            p.stride,
            p.padding,
            p.dilation,
            ds.device_ptr(),
            inp.device_ptr(),
            k,
            out.device_ptr(),
        );
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct ConvTranspose2D<'a>(&'a crate::conv::ParamsConvTranspose2D);
impl<'a> Map2 for ConvTranspose2D<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        inp: &GcuSlice<T>,
        inp_l: &Layout,
        k: &GcuSlice<T>,
        k_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>> {
        // Kernel shape: (c_in_k, c_out, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let p = &self.0;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let func = dev.get_or_load_func(&kernel_name::<T>("conv_transpose2d"), ubridge::CONV)?;
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        };
        let ds = dev.htod_copy(ds)?;
        let params = (
            el,
            out_w,
            out_h,
            p.stride,
            p.padding,
            p.output_padding,
            ds.device_ptr(),
            inp.device_ptr(),
            k.device_ptr(),
            out.device_ptr(),
        );
        // SAFETY: ffi.
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

enum PoolOp {
    Max,
    Avg,
}

struct Pool2D {
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    op: PoolOp,
}

impl Map1 for Pool2D {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        inp: &GcuSlice<T>,
        dev: &GcuDevice,
        inp_l: &Layout,
    ) -> Result<GcuSlice<T>> {
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for pool {dims:?}")
        };
        let el = shape.elem_count();
        let out_w = (dims[2] - self.w_k) / self.w_stride + 1;
        let out_h = (dims[3] - self.h_k) / self.h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let kname = match self.op {
            PoolOp::Max => "max_pool2d",
            PoolOp::Avg => "avg_pool2d",
        };
        let func = dev.get_or_load_func(&kernel_name::<T>(kname), ubridge::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let ds = dev.htod_copy(ds).w()?;
        let params = (
            el,
            self.w_k,
            self.h_k,
            self.w_stride,
            self.h_stride,
            ds.device_ptr(),
            inp.device_ptr(),
            out.device_ptr(),
        );
        // SAFETY: ffi.
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct UpsampleNearest2D(usize, usize);
impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        inp: &GcuSlice<T>,
        dev: &GcuDevice,
        inp_l: &Layout,
    ) -> Result<GcuSlice<T>> {
        // Input shape: (b_size, c, h, w)
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        };
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let func = dev.get_or_load_func(&kernel_name::<T>("upsample_nearest2d"), ubridge::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let ds = dev.htod_copy(ds).w()?;
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;
        let params = (
            out_w,
            out_h,
            scale_w,
            scale_h,
            ds.device_ptr(),
            inp.device_ptr(),
            out.device_ptr(),
        );
        // SAFETY: ffi.
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(out)
    }
}

struct WhereCond<'a>(&'a GcuStorage, &'a Layout);
impl<'a> Map2 for WhereCond<'a> {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        t: &GcuSlice<T>,
        layout_t: &Layout,
        f: &GcuSlice<T>,
        layout_f: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>> {
        let ids_l = &self.1;
        let shape = ids_l.shape();
        let el = shape.elem_count();
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);
        let out = dev.alloc::<T>(el).w()?;

        match &self.0.slice {
            GcuStorageSlice::U8(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("where_u8"), ubridge::TERNARY)?;
                let params = (
                    ptr.device_ptr(),
                    t.device_ptr(),
                    f.device_ptr(),
                    out.device_ptr(),
                    el,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            GcuStorageSlice::U32(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("where_u32"), ubridge::TERNARY)?;
                let params = (
                    ptr.device_ptr(),
                    t.device_ptr(),
                    f.device_ptr(),
                    out.device_ptr(),
                    el,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            GcuStorageSlice::I64(slice) => {
                let ptr = slice.slice(ids_l.start_offset()..);
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("where_i64"), ubridge::TERNARY)?;
                let params = (
                    ptr.device_ptr(),
                    t.device_ptr(),
                    f.device_ptr(),
                    out.device_ptr(),
                    el,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })?,
        };

        Ok(out)
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        lhs: &GcuSlice<T>,
        lhs_l: &Layout,
        rhs: &GcuSlice<T>,
        rhs_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<GcuSlice<T>> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs = &lhs.slice(lhs_l.start_offset()..);
        let rhs = &rhs.slice(rhs_l.start_offset()..);
        let out = dev.alloc::<T>(elem_count).w()?;
        let dims_and_strides = dev
            .htod_copy([dims, lhs_l.stride(), rhs_l.stride()].concat())
            .w()?;
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::BINARY)?;
        let params = (
            elem_count,
            dims.len(),
            dims_and_strides.device_ptr(),
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
        );
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;

        Ok(out)
    }
}

struct Cmp(CmpOp);
impl Map2Any for Cmp {
    fn f<T: DeviceCopy + WithDType>(
        &self,
        lhs: &GcuSlice<T>,
        lhs_l: &Layout,
        rhs: &GcuSlice<T>,
        rhs_l: &Layout,
        dev: &GcuDevice,
    ) -> Result<S> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let dims_and_strides = dev
            .htod_copy([dims, lhs_l.stride(), rhs_l.stride()].concat())
            .w()?;
        let lhs = &lhs.slice(lhs_l.start_offset()..);
        let rhs = &rhs.slice(rhs_l.start_offset()..);
        let name = match self.0 {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::BINARY)?;
        let out = dev.alloc::<u8>(elem_count).w()?;
        let params = (
            elem_count,
            dims.len(),
            dims_and_strides.device_ptr(),
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
        );
        unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
        Ok(S::U8(out))
    }
}

fn slice_src_and_dst<'a, T: DeviceCopy>(
    src: &'a GcuSlice<T>,
    src_l: &Layout,
    dst: &'a mut GcuSlice<T>,
    dst_offset: usize,
) -> (GcuView<'a, T>, GcuViewMut<'a, T>) {
    let src_offset = src_l.start_offset();
    let dst_copy = dst.len.saturating_sub(dst_offset);
    let src_copy = src.len.saturating_sub(src_offset);
    let src = src.slice(src_offset..src_offset + src_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + dst_copy);
    (src, dst)
}

pub trait GcuDType: Sized + DeviceCopy {
    fn as_gcu_slice(s: &GcuStorage) -> Result<&GcuSlice<Self>>;
    fn wrap_gcu_slice(s: GcuSlice<Self>, dev: GcuDevice) -> GcuStorage;
}

macro_rules! Gcu_dtype {
    ($ty:ty, $dtype:ident) => {
        impl GcuDType for $ty {
            fn as_gcu_slice(s: &GcuStorage) -> Result<&GcuSlice<Self>> {
                match &s.slice {
                    GcuStorageSlice::$dtype(data) => Ok(&data),
                    _ => Err(crate::Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn wrap_gcu_slice(slice: GcuSlice<Self>, device: GcuDevice) -> GcuStorage {
                let slice = GcuStorageSlice::$dtype(slice);
                GcuStorage { slice, device }
            }
        }
    };
}
Gcu_dtype!(u8, U8);
Gcu_dtype!(u32, U32);
Gcu_dtype!(i32, I32);
Gcu_dtype!(i64, I64);
Gcu_dtype!(f16, F16);
Gcu_dtype!(bf16, BF16);
Gcu_dtype!(f32, F32);
Gcu_dtype!(f64, F64);

impl GcuStorage {
    pub fn wrap_gcu_slice<T: GcuDType>(slice: GcuSlice<T>, device: GcuDevice) -> GcuStorage {
        T::wrap_gcu_slice(slice, device)
    }

    pub fn as_gcu_slice<T: GcuDType>(&self) -> Result<&GcuSlice<T>> {
        T::as_gcu_slice(self)
    }

    fn to_dtype_impl(&self, src: &Self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let el = shape.elem_count();
        let dev = self.device();
        let start_o = layout.start_offset();
        // This returns an i64 rather than a &i64, this is useful to get around some temporary
        // lifetime issue and is safe as long as self.slice does not go out of scope before inp
        // is used.
        let inp = match &src.slice {
            GcuStorageSlice::U8(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::I8(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::U32(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::I32(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::I64(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::BF16(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F16(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F32(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F64(inp) => inp.slice(start_o..).device_ptr(),
        };
        let inp = &inp;
        let kernel_name = format!("cast_{}_{}", src.dtype().as_str(), dtype.as_str());
        let mut cfg = dev.launch_cfg.clone();
        cfg.set_shared_memory(el as u32 * self.dtype().size_in_bytes() as u32);

        let func = dev.get_or_load_func(&kernel_name, ubridge::CAST)?;
        let slice = match dtype {
            DType::U8 => {
                let out = dev.device.alloc::<u8>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::U8(out)
            }
            DType::I8 => {
                let out = dev.device.alloc::<i8>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::I8(out)
            }
            DType::U32 => {
                let out = dev.alloc::<u32>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::U32(out)
            }
            DType::I32 => {
                let out = dev.alloc::<i32>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::I32(out)
            }
            DType::I64 => {
                let out = dev.alloc::<i64>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::I64(out)
            }
            DType::BF16 => {
                let out = dev.alloc::<bf16>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            DType::F16 => {
                let out = dev.alloc::<f16>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            DType::F32 => {
                let out = dev.alloc::<f32>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            DType::F64 => {
                let out = dev.alloc::<f64>(el).w()?;
                let params = (el, *inp, out.device_ptr());
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::F64(out)
            }
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }
}

impl BackendStorage for GcuStorage {
    type Device = GcuDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let slice = Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> DType {
        match self.slice {
            GcuStorageSlice::U8(_) => DType::U8,
            GcuStorageSlice::I8(_) => DType::I8,
            GcuStorageSlice::U32(_) => DType::U32,
            GcuStorageSlice::I32(_) => DType::I32,
            GcuStorageSlice::I64(_) => DType::I64,
            GcuStorageSlice::BF16(_) => DType::BF16,
            GcuStorageSlice::F16(_) => DType::F16,
            GcuStorageSlice::F32(_) => DType::F32,
            GcuStorageSlice::F64(_) => DType::F64,
        }
    }

    fn device(&self) -> &GcuDevice {
        &self.device
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if layout.is_contiguous() {
            self.to_dtype_impl(self, layout, dtype)
        } else {
            //cast op does not support non-contiguous operand
            let device = self.device().clone();
            let mut src_l = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
            self.copy_strided_src(&mut src_l, 0, layout)?; //convert to contiguous
            self.to_dtype_impl(&src_l, &Layout::contiguous(layout.shape()), dtype)
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        //TODO
        let device = self.device().clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on gcu")
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let src_shape = layout.shape();
        if sum_dims[0] != src_shape.dims().len() - 1 {
            //reduce at other dim (not last dim), requires transpose
            let src_stride = layout.stride();
            let src_dims = layout.shape().dims();
            let mut dims = vec![];
            let mut stride = vec![];
            for (dim_idx, &d) in src_dims.iter().enumerate() {
                if !sum_dims.contains(&dim_idx) {
                    dims.push(d);
                    stride.push(src_stride[dim_idx]);
                }
            }
            for &dim_idx in sum_dims.iter() {
                dims.push(src_dims[dim_idx]);
                stride.push(src_stride[dim_idx]);
            }

            //use copy op for transpose
            let mut dst_layout = Layout::new(dims.into(), stride, 0);
            dst_layout.backup.insert(0, layout.clone());
            dst_layout
                .transform_ops
                .insert(0, crate::layout::LayoutTransformOP::TransformTranspose);

            let mut src_l = unsafe { device.alloc_uninit(src_shape, self.dtype())? };
            self.copy_strided_src(&mut src_l, 0, &dst_layout)?; //convert to contiguous

            let slice = FastReduce(sum_dims, op).map(&src_l.slice, &device, layout)?;
            Ok(Self { slice, device })
        } else {
            let slice = FastReduce(sum_dims, op).map(&self.slice, &device, layout)?;
            Ok(Self { slice, device })
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let r_strides: usize = rhs_l.stride().iter().product();
        if rhs_l.is_contiguous() || (!rhs_l.is_contiguous() && r_strides == 0) {
            let slice = Cmp(op).map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
            Ok(Self { slice, device })
        } else {
            let mut src = unsafe { device.alloc_uninit(rhs_l.shape(), rhs.dtype())? };
            self.copy_strided_src(&mut src, 0, rhs_l)?;
            let slice = Cmp(op).map(
                &self.slice,
                lhs_l,
                &src.slice,
                &Layout::contiguous(rhs_l.shape()),
                &device,
            )?;
            Ok(Self { slice, device })
        }
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        if layout.is_contiguous() {
            let slice = U::V.map(&self.slice, &device, layout)?;
            Ok(Self { slice, device })
        } else {
            let mut src = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
            self.copy_strided_src(&mut src, 0, layout)?;
            let slice = U::V.map(&src.slice, &device, &Layout::contiguous(layout.shape()))?;
            Ok(Self { slice, device })
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        if lhs_l.is_contiguous() {
            if !rhs_l.is_contiguous() {
                let mut src_r = unsafe { device.alloc_uninit(rhs_l.shape(), self.dtype())? };
                rhs.copy_strided_src(&mut src_r, 0, rhs_l)?; //convert to contiguous
                let slice = B::V.map(
                    &self.slice,
                    lhs_l,
                    &src_r.slice,
                    &Layout::contiguous(rhs_l.shape()),
                    &device,
                )?;
                Ok(Self { slice, device })
            } else {
                let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
                Ok(Self { slice, device })
            }
        } else {
            //binary op does not support non-contiguous left operand
            let mut src_l = unsafe { device.alloc_uninit(lhs_l.shape(), self.dtype())? };
            self.copy_strided_src(&mut src_l, 0, lhs_l)?; //convert to contiguous
            if !rhs_l.is_contiguous() {
                let mut src_r = unsafe { device.alloc_uninit(rhs_l.shape(), self.dtype())? };
                rhs.copy_strided_src(&mut src_r, 0, rhs_l)?; //convert to contiguous
                let slice = B::V.map(
                    &src_l.slice,
                    &Layout::contiguous(lhs_l.shape()),
                    &src_r.slice,
                    &Layout::contiguous(rhs_l.shape()),
                    &device,
                )?;
                Ok(Self { slice, device })
            } else {
                let slice = B::V.map(
                    &src_l.slice,
                    &Layout::contiguous(lhs_l.shape()),
                    &rhs.slice,
                    rhs_l,
                    &device,
                )?;
                Ok(Self { slice, device })
            }
        }
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            GcuStorageSlice::U8(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::U8(cpu_storage))
            }
            GcuStorageSlice::I8(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::I8(cpu_storage))
            }
            GcuStorageSlice::U32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::U32(cpu_storage))
            }
            GcuStorageSlice::I32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::I32(cpu_storage))
            }
            GcuStorageSlice::I64(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::I64(cpu_storage))
            }
            GcuStorageSlice::BF16(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::BF16(cpu_storage))
            }
            GcuStorageSlice::F16(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::F16(cpu_storage))
            }
            GcuStorageSlice::F32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            GcuStorageSlice::F64(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::F64(cpu_storage))
            }
        }
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        if t_l.is_contiguous() && f_l.is_contiguous() {
            if layout.is_contiguous() {
                let slice = WhereCond(self, layout).map(&t.slice, t_l, &f.slice, f_l, &device)?;
                return Ok(Self { slice, device });
            } else {
                let mut src = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
                self.copy_strided_src(&mut src, 0, layout)?; //convert to contiguous
                let slice = WhereCond(&src, &Layout::contiguous(layout.shape()))
                    .map(&t.slice, t_l, &f.slice, f_l, &device)?;
                return Ok(Self { slice, device });
            }
        }

        if !t_l.is_contiguous() && !f_l.is_contiguous() {
            let mut src_t = unsafe { device.alloc_uninit(t_l.shape(), t.dtype())? };
            t.copy_strided_src(&mut src_t, 0, t_l)?; //convert to contiguous
            let mut src_f = unsafe { device.alloc_uninit(f_l.shape(), f.dtype())? };
            f.copy_strided_src(&mut src_f, 0, f_l)?; //convert to contiguous
            if layout.is_contiguous() {
                let slice = WhereCond(self, layout).map(
                    &src_t.slice,
                    &Layout::contiguous(t_l.shape()),
                    &src_f.slice,
                    &Layout::contiguous(f_l.shape()),
                    &device,
                )?;
                Ok(Self { slice, device })
            } else {
                let mut src = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
                self.copy_strided_src(&mut src, 0, layout)?; //convert to contiguous
                let slice = WhereCond(&src, &Layout::contiguous(layout.shape())).map(
                    &src_t.slice,
                    &Layout::contiguous(t_l.shape()),
                    &src_f.slice,
                    &Layout::contiguous(f_l.shape()),
                    &device,
                )?;
                Ok(Self { slice, device })
            }
        } else if !t_l.is_contiguous() {
            let mut src_t = unsafe { device.alloc_uninit(t_l.shape(), t.dtype())? };
            t.copy_strided_src(&mut src_t, 0, t_l)?; //convert to contiguous
            if layout.is_contiguous() {
                let slice = WhereCond(self, layout).map(
                    &src_t.slice,
                    &Layout::contiguous(t_l.shape()),
                    &f.slice,
                    f_l,
                    &device,
                )?;
                return Ok(Self { slice, device });
            } else {
                let mut src = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
                self.copy_strided_src(&mut src, 0, layout)?; //convert to contiguous
                let slice = WhereCond(&src, &Layout::contiguous(layout.shape())).map(
                    &src_t.slice,
                    &Layout::contiguous(t_l.shape()),
                    &f.slice,
                    f_l,
                    &device,
                )?;
                return Ok(Self { slice, device });
            }
        } else {
            let mut src_f = unsafe { device.alloc_uninit(f_l.shape(), f.dtype())? };
            f.copy_strided_src(&mut src_f, 0, f_l)?; //convert to contiguous
            if layout.is_contiguous() {
                let slice = WhereCond(self, layout).map(
                    &t.slice,
                    t_l,
                    &src_f.slice,
                    &Layout::contiguous(f_l.shape()),
                    &device,
                )?;
                return Ok(Self { slice, device });
            } else {
                let mut src = unsafe { device.alloc_uninit(layout.shape(), self.dtype())? };
                self.copy_strided_src(&mut src, 0, layout)?; //convert to contiguous
                let slice = WhereCond(&src, &Layout::contiguous(layout.shape())).map(
                    &t.slice,
                    t_l,
                    &src_f.slice,
                    &Layout::contiguous(f_l.shape()),
                    &device,
                )?;
                return Ok(Self { slice, device });
            }
        }
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = Conv1D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
        Ok(Self { slice, device })
    }

    #[cfg(not(feature = "cudnn"))]
    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = Conv2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
        Ok(Self { slice, device })
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice =
            ConvTranspose2D(params).map(&self.slice, l, &kernel.slice, kernel_l, &device)?;
        Ok(Self { slice, device })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Avg,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let slice = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Max,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = UpsampleNearest2D(out_w, out_h).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }
    fn scatter_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        ScatterAdd(ids, ids_l, dim).map(&mut acc.slice, l.shape(), &src.slice, src_l, &device)?;
        Ok(acc)
    }
    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (mut slice, l) = if l.is_contiguous() {
            (self.to_owned(), l)
        } else {
            let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
            self.copy_strided_src(&mut acc, 0, l)?;
            (acc, &Layout::contiguous(l.shape()))
        };
        if src_l.is_contiguous() {
            IndexAdd(ids, ids_l, dim).map(
                &mut slice.slice,
                l.shape(),
                &src.slice,
                src_l,
                &device,
            )?;
        } else {
            assert!(src_l.is_contiguous())
        }
        Ok(slice)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let mut lhs_transpose = 0;
        let mut rhs_transpose = 0;
        for i in 0..lhs_l.transform_ops.len() {
            let op: usize = lhs_l.transform_ops[i].to_owned().into();
            if op == 1 {
                lhs_transpose = 1;
                break;
            }
        }

        let mut broadcasted_weight: i32 = 0;
        for i in 0..rhs_l.transform_ops.len() {
            let op: usize = rhs_l.transform_ops[i].to_owned().into();
            if op == 2 {
                broadcasted_weight = 1;
            }
            if op == 1 {
                rhs_transpose = 1;
                break;
            }
        }

        let slice = match (&self.slice, &rhs.slice) {
            (GcuStorageSlice::BF16(lhs), GcuStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<bf16>(elem_count).w()?;
                // let bias = dev.alloc::<bf16>(n).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataBf16,
                    ubridge::DATATYPE::DataBf16,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    rhs_transpose,
                );

                let mut cfg = dev.launch_cfg.clone();
                cfg.set_shared_memory(
                    (lhs_l.shape().elem_count() as i32 + 2 * param.sip_k * param.sip_m) as u32 * 16,
                );
                let kernel_name = "matmul_bf16".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;
                let params = (
                    lhs.device_ptr(),
                    rhs.device_ptr(),
                    out.device_ptr(), //bias.device_ptr(),
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                );
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            (GcuStorageSlice::F16(lhs), GcuStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f16>(elem_count).w()?;
                // let bias = dev.alloc::<f16>(n).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataFp16,
                    ubridge::DATATYPE::DataFp16,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    rhs_transpose,
                );
                let mut cfg = dev.launch_cfg.clone();
                cfg.set_shared_memory(
                    (lhs_l.shape().elem_count() as i32 + 2 * param.sip_k * param.sip_m) as u32 * 4,
                );

                let kernel_name = "matmul_f16".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;

                let params = (
                    lhs.device_ptr(),
                    rhs.device_ptr(),
                    out.device_ptr(), //bias.device_ptr(),
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                );
                // println!("GEMM F16: [{} {}, {}, {}], SIP [{} {} {}]", b, m, k, n, param.sip_m, param.sip_k, param.sip_n);

                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            (GcuStorageSlice::F32(lhs), GcuStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f32>(elem_count).w()?;
                // let bias = dev.alloc::<f32>(n).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataFp32,
                    ubridge::DATATYPE::DataFp32,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    rhs_transpose,
                );

                let kernel_name = "matmul_f32".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;

                let params = (
                    lhs.device_ptr(),
                    rhs.device_ptr(),
                    out.device_ptr(), //bias.device_ptr(),
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                );
                // println!("GEMM F32: [{} {}, {}, {}], SIP [{} {} {}]", b, m, k, n, param.sip_m, param.sip_k, param.sip_n);
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            (GcuStorageSlice::F64(lhs), GcuStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f64>(elem_count).w()?;
                // let bias = dev.alloc::<f64>(n).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataF64,
                    ubridge::DATATYPE::DataF64,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    rhs_transpose,
                );
                let kernel_name = "matmul_f64".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;

                let params = (
                    lhs.device_ptr(),
                    rhs.device_ptr(),
                    out.device_ptr(), //bias.device_ptr(),
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    k,
                    n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                );
                unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
                GcuStorageSlice::F64(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in matmul op"))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    fn copy2d(
        &self,
        _dst: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_s: usize,
        _dst_s: usize,
        _src_o: usize,
        _dst_o: usize,
    ) -> Result<()> {
        todo!()
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        let origin_l = if !src_l.backup.is_empty() {
            &src_l.backup[0]
        } else {
            src_l
        };
        let origin_shape = origin_l.shape();
        let origin_el_count = origin_shape.elem_count();
        let dev = &self.device;
        let mut op_type: usize = 0;
        let mut dst_layout = (0..dims.len()).collect::<Vec<usize>>();

        if src_l.transform_ops.len() == 1 {
            op_type = src_l.transform_ops[0].into();
            if let Some(trans_dims) = &src_l.transpose_dims {
                dst_layout.swap(trans_dims[0], trans_dims[1]);
            }
        }
        //dst shape, dst stride, dst layout, origin shape
        let ds = dev
            .htod_copy([dims, src_l.stride(), &dst_layout, origin_shape.dims()].concat())
            .w()?;
        let mut cfg = dev.launch_cfg.clone();

        if (op_type == 1 && origin_el_count == el_count && dims.len() < 5)
            || (op_type == 2
                && origin_el_count < el_count
                && origin_shape.dims().len() <= dims.len())
            || (op_type == 4 && origin_el_count > el_count)
        {
        } else {
            let shared_memory_required =
                (origin_el_count + el_count) * self.dtype().size_in_bytes();
            cfg.set_shared_memory(shared_memory_required as u32);
        };

        match (&self.slice, &mut dst.slice) {
            (GcuStorageSlice::BF16(src), GcuStorageSlice::BF16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_bf16", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F16(src), GcuStorageSlice::F16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_f16", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F32(src), GcuStorageSlice::F32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_f32", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::U8(src), GcuStorageSlice::U8(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_u8", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::I8(src), GcuStorageSlice::I8(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_i8", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::U32(src), GcuStorageSlice::U32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_u32", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::I64(src), GcuStorageSlice::I64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_i64", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F64(src), GcuStorageSlice::F64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() && origin_el_count == el_count {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_64", ubridge::UNARY)?;
                    let params = (
                        origin_el_count,
                        el_count,
                        dims.len(),
                        origin_shape.dims().len(),
                        ds.device_ptr(),
                        src.device_ptr(),
                        dst.device_ptr(),
                        op_type,
                    );
                    unsafe { func.launch(&cfg, params) }.w()?;
                }
            }
            _ => Err(GcuError::InternalError("dtype mismatch in copy_strided op"))?,
        }
        Ok(())
    }
}

pub struct Rope {
    pub cos_sin_length: i32,
    pub cos_sin_stride: i32,
    pub index_positions: Vec<i32>,
    pub batch: i32,
    pub num_tokens: i32,
    pub q_head_size: i32,
    pub k_head_size: i32,
    pub hidden_size: i32,
    pub split_dim: i32,
    pub gpt_neox: i32,
}
impl crate::CustomOp3 for Rope {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str {
        "rope"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        crate::bail!("no cpu support for rope")
    }

    /// The forward pass, as run on a gcu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn gcu_fwd(
        &self,
        query: &GcuStorage,
        query_l: &Layout,
        key: &GcuStorage,
        _key_l: &Layout,
        cos_sin: &GcuStorage,
        _cos_sin_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        let dev = &query.device;
        let cfg = &dev.launch_cfg;

        let query_l = if !query_l.backup.is_empty() {
            &query_l.backup[0]
        } else {
            query_l
        };
        let shape = query_l.shape();

        let positions = dev.htod_copy(self.index_positions.to_vec()).w()?;

        match (&query.slice, &key.slice) {
            (GcuStorageSlice::BF16(query_), GcuStorageSlice::BF16(key_)) => {
                let (func, cos_sin_ptr) = match &cos_sin.slice {
                    GcuStorageSlice::BF16(cos_sin_) => (
                        dev.get_or_load_func("rope_bf16", ubridge::EMBEDDING)?,
                        cos_sin_.device_ptr(),
                    ),
                    GcuStorageSlice::F32(cos_sin_) => (
                        dev.get_or_load_func("rope_f32_bf16", ubridge::EMBEDDING)?,
                        cos_sin_.device_ptr(),
                    ),
                    _ => Err(GcuError::InternalError("dtype mismatch in rope op"))?,
                };
                let params = (
                    query_.device_ptr(),
                    key_.device_ptr(),
                    cos_sin_ptr,
                    self.cos_sin_length,
                    self.cos_sin_stride,
                    positions.device_ptr(),
                    self.batch,
                    self.num_tokens,
                    self.q_head_size,
                    self.k_head_size,
                    self.hidden_size,
                    self.split_dim,
                    self.gpt_neox,
                );
                unsafe { func.launch(cfg, params) }.w()?;
            }
            (GcuStorageSlice::F32(query_), GcuStorageSlice::F32(key_)) => {
                let (func, cos_sin_ptr) = match &cos_sin.slice {
                    GcuStorageSlice::F32(cos_sin_) => (
                        dev.get_or_load_func("rope_f32", ubridge::EMBEDDING)?,
                        cos_sin_.device_ptr(),
                    ),
                    _ => Err(GcuError::InternalError("dtype mismatch in rope op"))?,
                };
                let params = (
                    query_.device_ptr(),
                    key_.device_ptr(),
                    cos_sin_ptr,
                    self.cos_sin_length,
                    self.cos_sin_stride,
                    positions.device_ptr(),
                    self.batch,
                    self.num_tokens,
                    self.q_head_size,
                    self.k_head_size,
                    self.hidden_size,
                    self.split_dim,
                    self.gpt_neox,
                );
                unsafe { func.launch(cfg, params) }.w()?;
            }
            (GcuStorageSlice::F16(query_), GcuStorageSlice::F16(key_)) => {
                let (func, cos_sin_ptr) = match &cos_sin.slice {
                    GcuStorageSlice::F16(cos_sin_) => (
                        dev.get_or_load_func("rope_f16", ubridge::EMBEDDING)?,
                        cos_sin_.device_ptr(),
                    ),
                    GcuStorageSlice::F32(cos_sin_) => (
                        dev.get_or_load_func("rope_f32_f16", ubridge::EMBEDDING)?,
                        cos_sin_.device_ptr(),
                    ),
                    _ => Err(GcuError::InternalError("dtype mismatch in rope op"))?,
                };
                let params = (
                    query_.device_ptr(),
                    key_.device_ptr(),
                    cos_sin_ptr,
                    self.cos_sin_length,
                    self.cos_sin_stride,
                    positions.device_ptr(),
                    self.batch,
                    self.num_tokens,
                    self.q_head_size,
                    self.k_head_size,
                    self.hidden_size,
                    self.split_dim,
                    self.gpt_neox,
                );
                unsafe { func.launch(cfg, params) }.w()?;
            }
            _ => Err(GcuError::InternalError("dtype mismatch in rope op"))?,
        };

        let device = dev.clone();
        Ok((
            GcuStorage {
                slice: query.slice.to_owned(),
                device,
            },
            shape.to_owned(),
        ))
    }
}

pub struct KVConcat {
    pub concat_dim: i32,
}
impl crate::CustomOp2 for KVConcat {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str {
        "kvconcat"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        crate::bail!("no cpu support for kvconcat")
    }

    /// The forward pass, as run on a gcu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn gcu_fwd(
        &self,
        ltensor: &GcuStorage,
        ltensor_l: &Layout,
        rtensor: &GcuStorage,
        rtensor_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        assert!(self.concat_dim == 2 || self.concat_dim == 0 || self.concat_dim == 1); //must be in the dim of sequence len
        let dev = &ltensor.device;
        let cfg = &dev.launch_cfg;
        let elem_count = ltensor_l.shape().elem_count() + rtensor_l.shape().elem_count();
        let dims = ltensor_l.shape().dims().len();
        let ds = dev
            .htod_copy([ltensor_l.shape().dims(), rtensor_l.shape().dims()].concat())
            .w()?;
        let slice = match (&ltensor.slice, &rtensor.slice) {
            (GcuStorageSlice::BF16(left_), GcuStorageSlice::BF16(right_)) => {
                let out = dev.alloc::<bf16>(elem_count).w()?;
                let func = dev.get_or_load_func("kvconcat_bf16", ubridge::KCCONCAT)?;
                let params = (
                    left_.device_ptr(),
                    right_.device_ptr(),
                    out.device_ptr(),
                    ds.device_ptr(),
                    dims,
                    self.concat_dim,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            (GcuStorageSlice::F32(left_), GcuStorageSlice::F32(right_)) => {
                let out = dev.alloc::<f32>(elem_count).w()?;
                let func = dev.get_or_load_func("kvconcat_f32", ubridge::KCCONCAT)?;
                let params = (
                    left_.device_ptr(),
                    right_.device_ptr(),
                    out.device_ptr(),
                    ds.device_ptr(),
                    dims,
                    self.concat_dim,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            (GcuStorageSlice::F16(left_), GcuStorageSlice::F16(right_)) => {
                let out = dev.alloc::<f16>(elem_count).w()?;
                let func = dev.get_or_load_func("kvconcat_f16", ubridge::KCCONCAT)?;
                let params = (
                    left_.device_ptr(),
                    right_.device_ptr(),
                    out.device_ptr(),
                    ds.device_ptr(),
                    dims,
                    self.concat_dim,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            (GcuStorageSlice::F64(left_), GcuStorageSlice::F64(right_)) => {
                let out = dev.alloc::<f64>(elem_count).w()?;
                let func = dev.get_or_load_func("kvconcat_f64", ubridge::KCCONCAT)?;
                let params = (
                    left_.device_ptr(),
                    right_.device_ptr(),
                    out.device_ptr(),
                    ds.device_ptr(),
                    dims,
                    self.concat_dim,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F64(out)
            }
            (GcuStorageSlice::U8(left_), GcuStorageSlice::U8(right_)) => {
                let out = dev.alloc::<u8>(elem_count).w()?;
                let func = dev.get_or_load_func("kvconcat_u8", ubridge::KCCONCAT)?;
                let params = (
                    left_.device_ptr(),
                    right_.device_ptr(),
                    out.device_ptr(),
                    ds.device_ptr(),
                    dims,
                    self.concat_dim,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::U8(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in kvconcat op"))?,
        };

        let mut lshape: Vec<usize> = ltensor_l.shape().dims().to_vec();
        if self.concat_dim == 0 {
            lshape[0] += rtensor_l.shape().dims()[0];
        } else if self.concat_dim == 1 {
            lshape[1] += rtensor_l.shape().dims()[1];
        } else if dims > 3 {
            lshape[2] += rtensor_l.shape().dims()[2];
        } else {
            lshape[1] += rtensor_l.shape().dims()[1];
        }

        let device = dev.clone();
        Ok((GcuStorage { slice, device }, lshape.into()))
    }
}

pub struct LayerNorm {
    pub eps: f32,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    pub remove_mean: bool,
    pub affine: bool,
}
impl crate::CustomOp3 for LayerNorm {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str {
        "layernorm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        crate::bail!("no cpu support for layernorm")
    }

    /// The forward pass, as run on a gcu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn gcu_fwd(
        &self,
        x: &GcuStorage,
        x_l: &Layout,
        weight: &GcuStorage,
        _weight_l: &Layout,
        bias: &GcuStorage,
        _bias_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        let dev = &x.device;
        let cfg = &dev.launch_cfg;
        let elem_count = x_l.shape().elem_count();
        let dims = x_l.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let (batch, chunks, last_dim_size) = if dims.len() == 1 {
            (1, 1, dim_m1)
        } else {
            (dims[0], elem_count / dims[0] / dim_m1, dim_m1)
        };

        let slice = match (&x.slice, &weight.slice, &bias.slice) {
            (GcuStorageSlice::BF16(x_), GcuStorageSlice::BF16(w_), GcuStorageSlice::BF16(b_)) => {
                let out = dev.alloc::<bf16>(elem_count).w()?;
                let func = dev.get_or_load_func("layernorm_bf16", ubridge::REDUCE)?;
                let params = (
                    x_.device_ptr(),
                    out.device_ptr(),
                    w_.device_ptr(),
                    b_.device_ptr(),
                    batch as i32,
                    chunks as i32,
                    last_dim_size as i32,
                    self.eps,
                    self.remove_mean as i32,
                    self.affine as i32,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            (GcuStorageSlice::F32(x_), GcuStorageSlice::F32(w_), GcuStorageSlice::F32(b_)) => {
                let out = dev.alloc::<f32>(elem_count).w()?;
                let func = dev.get_or_load_func("layernorm_f32", ubridge::REDUCE)?;
                let params = (
                    x_.device_ptr(),
                    out.device_ptr(),
                    w_.device_ptr(),
                    b_.device_ptr(),
                    batch as i32,
                    chunks as i32,
                    last_dim_size as i32,
                    self.eps,
                    self.remove_mean as i32,
                    self.affine as i32,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            (GcuStorageSlice::F16(x_), GcuStorageSlice::F16(w_), GcuStorageSlice::F16(b_)) => {
                let out = dev.alloc::<f16>(elem_count).w()?;
                let func = dev.get_or_load_func("layernorm_f16", ubridge::REDUCE)?;
                let params = (
                    x_.device_ptr(),
                    out.device_ptr(),
                    w_.device_ptr(),
                    b_.device_ptr(),
                    batch as i32,
                    chunks as i32,
                    last_dim_size as i32,
                    self.eps,
                    self.remove_mean as i32,
                    self.affine as i32,
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in layernorm op"))?,
        };

        let device = dev.clone();
        Ok((GcuStorage { slice, device }, x_l.shape().into()))
    }
}

pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GeLU,
    Elu(f64),
    Silu,
}

impl crate::CustomOp1 for Activation {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str {
        match self {
            Activation::ReLU => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::GeLU => "gelu",
            Activation::Elu(_) => "elu",
            Activation::Silu => "silu",
        }
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        crate::bail!("no cpu support for gcu activation!")
    }

    fn gcu_fwd(&self, s: &GcuStorage, l: &Layout) -> Result<(GcuStorage, Shape)> {
        let dev = &s.device;
        let cfg = &dev.launch_cfg;
        let elem_count = l.shape().elem_count();
        let func_name = match self {
            Activation::ReLU => format!("urelu_{}", s.dtype().as_str()),
            Activation::Sigmoid => format!("usigmoid_{}", s.dtype().as_str()),
            Activation::Tanh => format!("utanh_{}", s.dtype().as_str()),
            Activation::GeLU => format!("ugelu_{}", s.dtype().as_str()),
            Activation::Silu => format!("usilu_{}", s.dtype().as_str()),
            Activation::Elu(_) => format!("uelu_{}", s.dtype().as_str()),
        };
        let func = dev.get_or_load_func(&func_name, ubridge::UNARY)?;

        let slice = match &s.slice {
            GcuStorageSlice::BF16(slice) => {
                let out = dev.alloc::<bf16>(elem_count).w()?;
                match self {
                    Activation::Elu(v) => {
                        let mut cfg = cfg.clone();
                        cfg.set_shared_memory(elem_count as u32 * s.dtype().size_in_bytes() as u32);
                        let params = (elem_count, slice.device_ptr(), out.device_ptr(), *v as f32);
                        unsafe { func.launch(&cfg, params) }.w()?;
                    }
                    _ => {
                        let params = (elem_count, slice.device_ptr(), out.device_ptr());
                        unsafe { func.launch(cfg, params) }.w()?;
                    }
                };
                GcuStorageSlice::BF16(out)
            }
            GcuStorageSlice::F16(slice) => {
                let out = dev.alloc::<f16>(elem_count).w()?;
                match self {
                    Activation::Elu(v) => {
                        let params = (elem_count, slice.device_ptr(), out.device_ptr(), *v as f32);
                        unsafe { func.launch(cfg, params) }.w()?;
                    }
                    _ => {
                        let params = (elem_count, slice.device_ptr(), out.device_ptr());
                        unsafe { func.launch(cfg, params) }.w()?;
                    }
                };
                GcuStorageSlice::F16(out)
            }
            GcuStorageSlice::F32(slice) => {
                let out = dev.alloc::<f32>(elem_count).w()?;
                match self {
                    Activation::Elu(v) => {
                        let params = (elem_count, slice.device_ptr(), out.device_ptr(), *v as f32);
                        unsafe { func.launch(cfg, params) }.w()?;
                    }
                    _ => {
                        let params = (elem_count, slice.device_ptr(), out.device_ptr());
                        unsafe { func.launch(cfg, params) }.w()?;
                    }
                };
                GcuStorageSlice::F32(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in activation op"))?,
        };

        let device = dev.clone();
        Ok((GcuStorage { slice, device }, l.shape().into()))
    }
}

pub struct GPTQMatMul {
    pub qzeros: Option<crate::Tensor>,
    pub g_idx: Option<crate::Tensor>,
    pub workspace: Option<crate::Tensor>,
    pub bits: i32,
    pub group_size: i32,
}

impl GPTQMatMul {
    fn gcu_fwd_t<T: GcuDType + DeviceCopy>(
        &self,
        x: &GcuStorage,
        x_l: &Layout,
        qweight: &GcuStorage,
        qweight_l: &Layout,
        scale: &GcuStorage,
        scale_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        let dev = qweight.device();
        let x_shape = x_l.dims();
        let weight_shape = qweight_l.dims();
        // let zero_shape = self.qzeros.shape().dims();
        // let scale_shape = scale_l.dims();

        // normally, for 4-bit quant, the pack factor is 8 (8 elements packed within an int32),
        // while, we repacked the weights to uint8 in gcu platform, therefore, the pack factor become 2 (2 elements packed within an uint8)
        let pack_factor = if self.bits == 4 {
            2 as usize
        } else {
            1 as usize
        };
        let marlin_format = self.workspace.is_some();
        let size_k =
            weight_shape[weight_shape.len() - 2] * pack_factor * if marlin_format { 2 } else { 1 }; //marlin format
        let size_n = weight_shape[weight_shape.len() - 1] / if marlin_format { 2 } else { 1 }; //marlin format

        if marlin_format && self.bits != 4 {
            panic!("marlin format is only used for 4-bit quantization under GCU platform!");
        }
        let mut out_shape: Vec<usize> = x_shape.to_vec();
        out_shape[x_shape.len() - 1] = size_n;
        let oshape: Shape = out_shape.into();
        let (b, m) = if x_shape.len() > 2 {
            (x_shape[0], x_shape[1])
        } else {
            (1, x_shape[0])
        };

        let elem_count = oshape.elem_count();
        let mut lhs_transpose = 0;
        let mut rhs_transpose = 0;
        for i in 0..x_l.transform_ops.len() {
            let op: usize = x_l.transform_ops[i].to_owned().into();
            if op == 1 {
                lhs_transpose = 1;
                break;
            }
        }

        let mut broadcasted_weight: i32 = 0;
        for i in 0..qweight_l.transform_ops.len() {
            let op: usize = qweight_l.transform_ops[i].to_owned().into();
            if op == 2 {
                broadcasted_weight = 1;
            }
            if op == 1 {
                rhs_transpose = 1;
                break;
            }
        }

        let slice = match (&x.slice, &scale.slice) {
            (GcuStorageSlice::BF16(lhs), GcuStorageSlice::BF16(sc)) => {
                let (func, rhs_ptr, wtype) = match &qweight.slice {
                    GcuStorageSlice::I8(rhs) => {
                        let kernel_name = "matmul_bf16_8bit".to_string();
                        let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;
                        let rhs = &rhs.slice(qweight_l.start_offset()..);
                        (func, rhs.device_ptr(), ubridge::DATATYPE::DataI8)
                    }
                    GcuStorageSlice::U8(rhs) => {
                        let kernel_name = "matmul_bf16_4bit".to_string();
                        let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;
                        let rhs = &rhs.slice(qweight_l.start_offset()..);
                        (func, rhs.device_ptr(), ubridge::DATATYPE::DataI4)
                    }
                    _ => Err(GcuError::InternalError("dtype mismatch in qmatmul op"))?,
                };
                let lhs = &lhs.slice(x_l.start_offset()..);
                let sc = &sc.slice(scale_l.start_offset()..);
                let qzeros_ptr = if self.qzeros.is_some() {
                    let (qzeros, qzeros_l) = self.qzeros.as_ref().unwrap().storage_and_layout();
                    let qzeros = match &*qzeros {
                        crate::Storage::Gcu(p) => p,
                        _ => panic!("qzeros must be a gcu tensor"),
                    };
                    let qzeros_ = qzeros.as_gcu_slice::<T>()?;
                    let qzeros_ = qzeros_.slice(qzeros_l.start_offset()..);
                    qzeros_.device_ptr()
                } else {
                    sc.device_ptr()
                };
                let out = dev.alloc::<bf16>(elem_count).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataBf16,
                    wtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    size_k,
                    size_n,
                    rhs_transpose,
                );
                let mut cfg = dev.launch_cfg.clone();
                cfg.set_shared_memory(
                    (x_l.shape().elem_count() as i32 * 2 + 12 * 2 * param.sip_k * param.sip_n)
                        as u32
                        * 4,
                );

                let params = (
                    lhs.device_ptr(),
                    rhs_ptr,
                    out.device_ptr(),
                    sc.device_ptr(),
                    qzeros_ptr,
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    size_k,
                    size_n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                    self.group_size,
                );
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            (GcuStorageSlice::F16(lhs), GcuStorageSlice::F16(sc)) => {
                let (func, rhs_ptr, wtype) = match &qweight.slice {
                    GcuStorageSlice::I8(rhs) => {
                        let kernel_name = "matmul_f16_8bit".to_string();
                        let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;
                        let rhs = &rhs.slice(qweight_l.start_offset()..);
                        (func, rhs.device_ptr(), ubridge::DATATYPE::DataI8)
                    }
                    GcuStorageSlice::U8(rhs) => {
                        let kernel_name = "matmul_f16_4bit".to_string();
                        let func = dev.get_or_load_func(&kernel_name, ubridge::MATMUL)?;
                        let rhs = &rhs.slice(qweight_l.start_offset()..);
                        (func, rhs.device_ptr(), ubridge::DATATYPE::DataI4)
                    }
                    _ => Err(GcuError::InternalError("dtype mismatch in qmatmul op"))?,
                };
                let lhs = &lhs.slice(x_l.start_offset()..);
                let sc = &sc.slice(scale_l.start_offset()..);
                let qzeros_ptr = if self.qzeros.is_some() {
                    let (qzeros, qzeros_l) = self.qzeros.as_ref().unwrap().storage_and_layout();
                    let qzeros = match &*qzeros {
                        crate::Storage::Gcu(p) => p,
                        _ => panic!("qzeros must be a gcu tensor"),
                    };
                    let qzeros_ = qzeros.as_gcu_slice::<T>()?;
                    let qzeros_ = qzeros_.slice(qzeros_l.start_offset()..);
                    qzeros_.device_ptr()
                } else {
                    sc.device_ptr()
                };

                let out = dev.alloc::<f16>(elem_count).w()?;
                // let bias = dev.alloc::<bf16>(n).w()?;
                let param = dev.get_gemm_launch_params(
                    ubridge::DATATYPE::DataFp16,
                    wtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    size_k,
                    size_n,
                    rhs_transpose,
                );
                let mut cfg = dev.launch_cfg.clone();
                cfg.set_shared_memory(
                    (x_l.shape().elem_count() as i32 + 12 * 2 * param.sip_k * param.sip_m) as u32
                        * 2,
                );
                let params = (
                    lhs.device_ptr(),
                    rhs_ptr,
                    out.device_ptr(),
                    sc.device_ptr(),
                    qzeros_ptr,
                    param.input_dtype,
                    if broadcasted_weight > 0 { 1 } else { b },
                    if broadcasted_weight > 0 { b * m } else { m },
                    size_k,
                    size_n,
                    param.lhs_multicore,
                    param.rhs_multicore,
                    param.batch_multicore,
                    lhs_transpose,
                    rhs_transpose,
                    param.alpha,
                    param.beta,
                    param.addmm_beta, //param.bias,
                    param.sip_m,
                    param.sip_k,
                    param.sip_n,
                    broadcasted_weight,
                    self.group_size,
                );
                unsafe { func.launch(&cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in qmatmul op"))?,
        };
        Ok((
            GcuStorage {
                slice,
                device: dev.clone(),
            },
            oshape,
        ))
    }
}

impl crate::CustomOp3 for GPTQMatMul {
    fn name(&self) -> &'static str {
        "GPTQMatMul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        crate::bail!("no cpu support for GPTQMatMul")
    }

    fn gcu_fwd(
        &self,
        x: &GcuStorage,
        x_l: &Layout,
        qweight: &GcuStorage,
        qweight_l: &Layout,
        scale: &GcuStorage,
        scale_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.gcu_fwd_t::<f16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::BF16 => self.gcu_fwd_t::<bf16>(x, x_l, qweight, qweight_l, scale, scale_l),
            dt => crate::bail!("GPTQMatMul is only supported for f16 and bf16 ({dt:?})"),
        }
    }
}

pub struct GPTQRepack {
    pub bits: i32,
}

impl GPTQRepack {
    fn gcu_fwd_t<T: GcuDType + DeviceCopy>(
        &self,
        qweight: &GcuStorage,
        qweight_l: &Layout,
    ) -> Result<(GcuStorage, Shape)> {
        let dev = qweight.device();
        let q_shape = qweight_l.dims();
        let mut out_shape: Vec<usize> = q_shape.to_vec();
        out_shape[0] = (q_shape[0] / 2) as usize;
        out_shape[1] = (q_shape[1] * 2) as usize;

        let oshape: Shape = out_shape.into();

        // Get gcu slices for all tensors
        let q = qweight.as_gcu_slice::<u32>()?;

        // Get gcu views for all tensors
        let q = q.slice(qweight_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = dev.alloc::<u32>(elem_count).w()?;

        // let out_ptr = out.device_ptr() as *const core::ffi::c_void;
        // let q_ptr = q.device_ptr() as *const core::ffi::c_void;

        let out = GcuStorage::wrap_gcu_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl crate::CustomOp1 for GPTQRepack {
    fn name(&self) -> &'static str {
        "GPTQRepack"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        super::bail!("no cpu support for GPTQRepack")
    }

    fn gcu_fwd(&self, qweight: &GcuStorage, qweight_l: &Layout) -> Result<(GcuStorage, Shape)> {
        match qweight.dtype() {
            DType::U32 => self.gcu_fwd_t::<u32>(qweight, qweight_l),
            dt => crate::bail!("GPTQRepack is only supported for i32/u32 weight ({dt:?})"),
        }
    }
}
