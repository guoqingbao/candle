use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
use half::{bf16, f16};
use ubridge::prelude::DevicePtr;
use uhal::memory::DevicePointerTrait;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
pub use cust_core::_hidden::{DeviceCopy};
use uhal::error::{DeviceError};
use std::time::{Duration, Instant};
use ubridge::gcu_device::GcuDevice as RawDevice;
use ubridge::gcu_device::{GcuFunction};
use ubridge::gcu_slice::{GcuSlice, GcuView, GcuViewMut};
use ubridge::gcu_launch::{GcuLaunchConfig, GcuLaunchAsync};
use ubridge::{*};

#[derive(Debug, Clone)]
enum GcuStorageSlice {
    U8(GcuSlice<u8>),
    U32(GcuSlice<u32>),
    I64(GcuSlice<i64>),
    BF16(GcuSlice<bf16>),
    F16(GcuSlice<f16>),
    F32(GcuSlice<f32>),
    F64(GcuSlice<f64>),
}

type S = GcuStorageSlice;

#[derive(Debug, Clone)]
pub struct GcuStorage {
    slice: GcuStorageSlice,
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

    #[error("{Gcu} when loading {module_name}")]
    Load {
        Gcu: String,
        module_name: String,
    },
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
        let cfg = GcuLaunchConfig::for_num_elems(elem_count as u32);
        let slice = match dtype {
            DType::U8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<u8>(elem_count).w()?;
                let func = self.get_or_load_func("fill_u8", ubridge::FILL)?;
                let params = (&data, v as u8, elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::U8(data)
            }
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<u32>(elem_count).w()?;
                let func = self.get_or_load_func("fill_u32", ubridge::FILL)?;
                let params = (&data, v as u32, elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::U32(data)
            }
            DType::I64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<i64>(elem_count).w()?;
                let func = self.get_or_load_func("fill_i64", ubridge::FILL)?;
                let params = (&data, v as i64, elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::I64(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.alloc::<bf16>(elem_count).w()?;
                let func = self.get_or_load_func("fill_bf16", ubridge::FILL)?;
                let params = (&data, bf16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f16>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f16", ubridge::FILL)?;
                let params = (&data, f16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f32>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f32", ubridge::FILL)?;
                let params = (&data, v as f32, elem_count);
                unsafe { func.launch(cfg, params).w()? };
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = self.device.alloc::<f64>(elem_count).w()?;
                let func = self.get_or_load_func("fill_f64", ubridge::FILL)?;
                let params = (&data, v, elem_count);
                unsafe { func.launch(cfg, params).w()? };
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
}

impl BackendDevice for GcuDevice {
    type Storage = GcuStorage;
    fn new(ordinal: usize) -> Result<Self>
    {
        Ok(GcuDevice {
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
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count).w()?;
                GcuStorageSlice::U32(data)
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

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<GcuStorage> {
        let elem_count = shape.elem_count();
        // let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // Gcurc changes.
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(GcuError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                ?
            }
            DType::F32 => {
                let mut data = self.device.alloc::<f32>(elem_count).w()?;
                // curand.0.fill_with_uniform(&mut data)?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = self.device.alloc::<f64>(elem_count).w()?;
                // curand.0.fill_with_uniform(&mut data)?;
                GcuStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            let layout = Layout::contiguous(shape);
            Affine(up - lo, lo).map(&slice, &self, &layout)?
        };
        Ok(GcuStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<GcuStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // Gcurc changes.
        let elem_count = shape.elem_count();
        // let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(GcuError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                ?
            }
            DType::F32 => {
                let mut data = self.device.alloc::<f32>(elem_count).w()?;
                // curand
                //     .0
                //     .fill_with_normal(&mut data, mean as f32, std as f32)
                //     ?;
                GcuStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = self.device.alloc::<f64>(elem_count).w()?;
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

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<GcuStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                GcuStorageSlice::U32(data)
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
}



trait Map1 {
    fn f<T: DeviceCopy + WithDType>( //+ValidAsZeroBits
        
        &self,
        src: &GcuSlice<T>,
        dev: &GcuDevice,
        layout: &Layout,
    ) -> Result<GcuSlice<T>>;

    fn map(&self, s: &S, d: &GcuDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
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
    fn f<T: DeviceCopy + WithDType>( //+ValidAsZeroBits
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
    fn f<T: DeviceCopy + WithDType>( //+ValidAsZeroBits
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
            S::U32(s) => self.f(s, d, l, S::U32)?,
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
    fn f<T: DeviceCopy>(
        &self,
        s: &GcuSlice<T>,
        _: &GcuDevice,
        _: &Layout,
    ) -> Result<GcuSlice<T>> {
        s.try_clone().w()
    }
}

fn kernel_name<T: WithDType>(root: &str) -> String {
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
        // let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        // let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("affine"), ubridge::AFFINE)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el)?;
        let params = (
            el,
            src,
            &out,
            self.0 as f32,
            self.1 as f32
            // T::from_f64(self.0),
            // T::from_f64(self.1),
        );
        unsafe { func.launch(cfg, params) }?;
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
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("uelu"), ubridge::UNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el)?;
        let params = (el, dims.len(), &ds, T::from_f64(self.0), src, &out);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }?;
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
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        let ds = dev
            .htod_copy([src_dims, layout.stride(), &sum_dims_l, &sum_dims_s].concat())
            ?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>("sum"), ubridge::REDUCE)?;
        let out = dev.alloc_zeros::<T>(dst_el).w()?;
        let params = (el, src_dims.len(), sum_dims.len(), &ds, src, &out);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
        let cfg = GcuLaunchConfig {
            // TODO: Maybe use grid_y if the output is too large?
            // TODO: Specialized implementation when reducing on no or all dimensions or when
            // reducing only aggregate a small number of elements together.
            grid_dim: (dst_el as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let ds = dev
            .htod_copy([dims.as_slice(), stride.as_slice()].concat()).w()
            ?;
        let src = &src.slice(layout.start_offset()..);
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
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::REDUCE)?;
        if return_index {
            // SAFETY: filled in by the follow up kernel.
            let out = dev.alloc::<u32>(dst_el).w()?;
            let params = (src_el, el_to_sum_per_block, src_dims.len(), &ds, src, &out);
            // SAFETY: ffi.
            unsafe { func.launch(cfg, params) }.w()?;
            Ok(S::U32(out))
        } else {
            // SAFETY: filled in by the follow up kernel.
            let out = dev.alloc::<T>(dst_el).w()?;
            let params = (src_el, el_to_sum_per_block, src_dims.len(), &ds, src, &out);
            // SAFETY: ffi.
            unsafe { func.launch(cfg, params) }.w()?;
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
        // let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el_count as u32);
        // let ds = dev.htod_copy([dims, layout.stride()].concat()).w()?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::UNARY)?;
        let out = dev.alloc::<T>(el_count).w()?;
        let params = (el_count, src, &out);
        unsafe { func.launch(cfg, params) }.w()?;
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
        let name = match &self.0.slice {
            GcuStorageSlice::U32(_) => {
                // ("is_u32", unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()})
                "is_u32"
            }
            GcuStorageSlice::U8(_) => {
                // ("is_u8", unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()})
                "is_u8"
            }
            GcuStorageSlice::I64(_) => {
                // ("is_i64", unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()})
                "is_i64"
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "index_select ids should be u8 or u32",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            ?,
        };
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ids_el = ids_shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(ids_el as u32);
        let ds = dev.htod_copy([ids_dims, ids_l.stride()].concat())?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let dim_size = src_l.dims()[self.2];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(ids_el * left_size * right_size).w()?;
        let params = (
            ids_el,
            ids_dims.len(),
            &ds,
            // ids, //TODO
            &src,
            &out,
            left_size,
            dim_size,
            right_size,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        // let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
        //     Some(o12) => o12,
        //     None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        // };
        let name = match &ids.slice {
            GcuStorageSlice::U32(_) => {
                // ("gather_u32", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()})
                "gather_u32"
            }
            GcuStorageSlice::U8(_) => {
                // ("gather_u8", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "gather_u8"
            }
            GcuStorageSlice::I64(_) => {
                // ("gather_i64", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()})
                "gather_i64"
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let el = ids_l.shape().elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el).w()?;
        let params = (
            el, 
            // ids, 
            &src, &out, left_sz, src_dim_sz, ids_dim_sz, right_sz,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        // let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
        //     Some(o12) => o12,
        //     None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        // };
        let name = match &ids.slice {
            GcuStorageSlice::U32(_) => {
                // ("ia_u32", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "ia_u32"
            }
            GcuStorageSlice::I64(_) => {
                // ("ia_i64", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "ia_i64"
            }
            GcuStorageSlice::U8(_) => {
                // ("ia_u8", unsafe {*slice.slice(ids_o1..ids_o2).device_ptr()}),
                "ia_u8"
            }
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
        let cfg = GcuLaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let params = (
            // ids, 
            ids_dim_sz, &src, dst.buffer.as_device_ptr(), left_sz, src_dim_sz, dst_dim_sz, right_sz,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems((left_sz * right_sz) as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::INDEXING)?;
        // SAFETY: Set later by running the kernel.
        let params = (
            // ids, 
            &src, dst.buffer.as_device_ptr(), left_sz, src_dim_sz, dst_dim_sz, right_sz);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let inp = &inp.slice(inp_l.start_offset()..);
        let k = &k.slice(k_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let cfg = GcuLaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv1d"), ubridge::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let ds = if dims.len() == 3 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else if dims.len() == 2 {
            [&[1], dims, &[1], inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        };
        let ds = dev.htod_copy(ds).w()?;
        let params = (el, l_out, p.stride, p.padding, &ds, inp, k, &out);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("conv2d"), ubridge::CONV)?;
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride(), k_l.dims(), k_l.stride()].concat()
        } else {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        };
        let ds = dev.htod_copy(ds).w()?;
        let params = (el, out_w, out_h, p.stride, p.padding, &ds, inp, k, &out);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems(dst_el as u32);
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
            &ds,
            inp,
            k,
            &out,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems(dst_el as u32);
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
            &ds,
            inp,
            &out,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("upsample_nearest2d"), ubridge::CONV)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(dst_el).w()?;
        let ds = dev.htod_copy(ds).w()?;
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;
        let params = (out_w, out_h, scale_w, scale_h, &ds, inp, &out);
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
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
        let name = match &self.0.slice {
            GcuStorageSlice::U8(_) => {
                // let ptr = unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()};
                // (ptr, "where_u8")
                "where_u8"
            }
            GcuStorageSlice::U32(_) => {
                // let ptr = unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()};
                // (ptr, "where_u32")
                "where_u32"
            }
            GcuStorageSlice::I64(_) => {
                // let ptr = unsafe {*slice.slice(ids_l.start_offset()..).device_ptr()};
                // (ptr, "where_i64")
                "where_i64"
            }
            _ => Err(GcuError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: self.0.dtype(),
            })
            ?,
        };
        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        let ds = dev
            .htod_copy([dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat()).w()
            ?;
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::TERNARY)?;
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<T>(el).w()?;
        let params = (el, dims.len(), &ds, 
            // ids, 
            t, f, &out);
        // SAFETY: ffi
        unsafe { func.launch(cfg, params) }.w()?;
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
        // let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(elem_count as u32);
        // let dims_and_strides = dev
        //     .htod_copy([dims, lhs_l.stride(), rhs_l.stride()].concat())
        //     ?;
        let lhs = &lhs.slice(lhs_l.start_offset()..);
        let rhs = &rhs.slice(rhs_l.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), ubridge::BINARY)?;
        let out = dev.alloc::<T>(elem_count).w()?;
        let params = (elem_count, lhs, rhs, &out);
        unsafe { func.launch(cfg, params) }.w()?;
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
        let cfg = GcuLaunchConfig::for_num_elems(elem_count as u32);
        let dims_and_strides = dev
            .htod_copy([dims, lhs_l.stride(), rhs_l.stride()].concat()).w()
            ?;
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
        // SAFETY: Set later by running the kernel.
        let out = dev.alloc::<u8>(elem_count).w()?;
        let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
        // SAFETY: ffi
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(S::U8(out))
    }
}

fn slice_src_and_dst<'a, T: DeviceCopy>(
    src: &'a GcuSlice<T>,
    src_l: &Layout,
    dst: &'a mut GcuSlice<T>,
    dst_offset: usize,
) -> (
    GcuView<'a, T>, //TODO
    GcuViewMut<'a, T>,
) {
    let src_offset = src_l.start_offset();
    let to_copy = dst
        .len
        .saturating_sub(dst_offset)
        .min(src.len.saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}

pub trait GcuDType: Sized + DeviceCopy {
    fn as_Gcu_slice(s: &GcuStorage) -> Result<&GcuSlice<Self>>;
    fn wrap_Gcu_slice(s: GcuSlice<Self>, dev: GcuDevice) -> GcuStorage;
}

macro_rules! Gcu_dtype {
    ($ty:ty, $dtype:ident) => {
        impl GcuDType for $ty {
            fn as_Gcu_slice(s: &GcuStorage) -> Result<&GcuSlice<Self>> {
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

            fn wrap_Gcu_slice(slice: GcuSlice<Self>, device: GcuDevice) -> GcuStorage {
                let slice = GcuStorageSlice::$dtype(slice);
                GcuStorage { slice, device }
            }
        }
    };
}
Gcu_dtype!(u8, U8);
Gcu_dtype!(u32, U32);
Gcu_dtype!(i64, I64);
Gcu_dtype!(f16, F16);
Gcu_dtype!(bf16, BF16);
Gcu_dtype!(f32, F32);
Gcu_dtype!(f64, F64);

impl GcuStorage {
    pub fn wrap_Gcu_slice<T: GcuDType>(slice: GcuSlice<T>, device: GcuDevice) -> GcuStorage {
        T::wrap_Gcu_slice(slice, device)
    }

    pub fn as_Gcu_slice<T: GcuDType>(&self) -> Result<&GcuSlice<T>> {
        T::as_Gcu_slice(self)
    }
}

// fn gemm_config<T>(
//     alpha: T,
//     beta: T,
//     (b, m, n, k): (usize, usize, usize, usize),
//     lhs_l: &Layout,
//     rhs_l: &Layout,
// ) -> DeviceResult<StridedBatchedConfig<T>> {
//     // https://docs.nvidia.com/Gcu/cublas/index.html#cublas-t-gemm
//     use Gcurc::cublas::sys::cublasOperation_t;

//     let lhs_stride = lhs_l.stride();
//     let rhs_stride = rhs_l.stride();
//     let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
//     let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
//     let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
//     let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
//     // The a tensor has dims batching, k, n (rhs)
//     let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
//         (n as i32, cublasOperation_t::CUBLAS_OP_N)
//     } else if rhs_m1 == k && rhs_m2 == 1 {
//         (k as i32, cublasOperation_t::CUBLAS_OP_T)
//     } else {
//         Err(GcuError::MatMulNonContiguous {
//             lhs_stride: lhs_stride.to_vec(),
//             rhs_stride: rhs_stride.to_vec(),
//             mnk: (m, n, k),
//         })?
//     };
//     // The b tensor has dims batching, m, k (lhs)
//     let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
//         (k as i32, cublasOperation_t::CUBLAS_OP_N)
//     } else if lhs_m1 == m && lhs_m2 == 1 {
//         (m as i32, cublasOperation_t::CUBLAS_OP_T)
//     } else {
//         Err(GcuError::MatMulNonContiguous {
//             lhs_stride: lhs_stride.to_vec(),
//             rhs_stride: rhs_stride.to_vec(),
//             mnk: (m, n, k),
//         })?
//     };
//     // The setup below was copied from:
//     // https://github.com/lebedov/scikit-Gcu/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
//     let gemm = GemmConfig {
//         alpha,
//         beta,
//         m: n as i32,
//         n: m as i32,
//         k: k as i32,
//         lda,
//         ldb,
//         ldc: n as i32,
//         transa,
//         transb,
//     };

//     let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
//         [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
//         [stride] => stride,
//         [] => m * k,
//         _ => Err(GcuError::MatMulNonContiguous {
//             lhs_stride: lhs_stride.to_vec(),
//             rhs_stride: rhs_stride.to_vec(),
//             mnk: (m, n, k),
//         })?,
//     };
//     let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
//         [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
//         [stride] => stride,
//         [] => n * k,
//         _ => Err(GcuError::MatMulNonContiguous {
//             lhs_stride: lhs_stride.to_vec(),
//             rhs_stride: rhs_stride.to_vec(),
//             mnk: (m, n, k),
//         })?,
//     };

//     Ok(StridedBatchedConfig {
//         batch_size: b as i32,
//         gemm,
//         stride_a: stride_a as i64,
//         stride_b: stride_b as i64,
//         stride_c: (m * n) as i64,
//     })
// }

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
            GcuStorageSlice::U32(_) => DType::U32,
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
        let shape = layout.shape();
        // let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        // let ds = dev.htod_copy([dims, layout.stride()].concat()).w()?;
        let start_o = layout.start_offset();
        // This returns an i64 rather than a &i64, this is useful to get around some temporary
        // lifetime issue and is safe as long as self.slice does not go out of scope before inp
        // is used.
        let inp = match &self.slice {
            GcuStorageSlice::U8(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::U32(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::I64(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::BF16(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F16(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F32(inp) => inp.slice(start_o..).device_ptr(),
            GcuStorageSlice::F64(inp) => inp.slice(start_o..).device_ptr(),
        };
        let inp = &inp;

        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
        let func = dev.get_or_load_func(&kernel_name, ubridge::CAST)?;
        let slice = match dtype {
            DType::U8 => {
                let out = dev.device.alloc::<u8>(el).w()?;
                let params = (el,  
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::U8(out)
            }
            DType::U32 => {
                let out = dev.alloc::<u32>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::U32(out)
            }
            DType::I64 => {
                let out = dev.alloc::<i64>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::I64(out)
            }
            DType::BF16 => {
                let out = dev.alloc::<bf16>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            DType::F16 => {
                let out = dev.alloc::<f16>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            DType::F32 => {
                let out = dev.alloc::<f32>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            DType::F64 => {
                let out = dev.alloc::<f64>(el).w()?;
                let params = (el, 
                    *inp, 
                    &out);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F64(out)
            }
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
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

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = Cmp(op).map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let slice = U::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = B::V.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?;
        Ok(Self { slice, device })
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            GcuStorageSlice::U8(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::U8(cpu_storage))
            }
            GcuStorageSlice::U32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice).w()?;
                Ok(CpuStorage::U32(cpu_storage))
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
        let slice = WhereCond(self, layout).map(&t.slice, t_l, &f.slice, f_l, &device)?;
        Ok(Self { slice, device })
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

    #[cfg(feature = "cudnn")]
    fn conv2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        if !kernel_l.is_contiguous() {
            let slice = Conv2D(params).map(&self.slice, inp_l, &kernel.slice, kernel_l, &device)?;
            return Ok(Self { slice, device });
        }
        let (out_w, out_h) = (params.out_w(), params.out_h());
        let dst_el = params.c_out * out_w * out_h * params.b_size;
        let slice = match (&self.slice, &kernel.slice) {
            (S::U8(inp), S::U8(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<u8>(dst_el) }?;
                crate::cudnn::launch_conv2d::<u8>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::U8(out)
            }
            (S::BF16(inp), S::BF16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<bf16>(dst_el) }?;
                crate::cudnn::launch_conv2d::<bf16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::BF16(out)
            }
            (S::F16(inp), S::F16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f16>(dst_el) }?;
                crate::cudnn::launch_conv2d::<f16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F16(out)
            }
            (S::F32(inp), S::F32(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f32>(dst_el) }?;
                crate::cudnn::launch_conv2d::<f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F32(out)
            }
            (S::F64(inp), S::F64(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f64>(dst_el) }?;
                crate::cudnn::launch_conv2d::<f64>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F64(out)
            }
            (S::U32(_), S::U32(_)) => Err(GcuError::InternalError("conv2d does not support u32"))?,
            (S::I64(_), S::I64(_)) => Err(GcuError::InternalError("conv2d does not support i64"))?,
            _ => Err(GcuError::InternalError("dtype mismatch in conv2d"))?,
        };
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
        let mut acc = device.zeros_impl(l.shape(), self.dtype())?;
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
        let mut acc = device.zeros_impl(l.shape(), self.dtype())?;
        self.copy_strided_src(&mut acc, 0, l)?;
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, l.shape(), &src.slice, src_l, &device)?;
        Ok(acc)
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

        let slice = match (&self.slice, &rhs.slice) {
            (GcuStorageSlice::BF16(lhs), GcuStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<bf16>(elem_count).w()?;
                let bias = dev.alloc::<bf16>(n).w()?;
                let param = dev.get_gemm_launch_params(ubridge::DATATYPE::DataBf16, b, m, k, n);
                let kernel_name = "gemm_bf16".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::GEMM)?;
                let cfg = GcuLaunchConfig::for_gemm();
                let params = (lhs, rhs, &out, &bias, 
                    param.input_dtype, b, m, k, n,
                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                    param.lhs_transpose, param.rhs_transpose,
                    param.alpha, param.beta, param.addmm_beta, param.bias,
                    param.sip_m, param.sip_k, param.sip_n
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::BF16(out)
            }
            (GcuStorageSlice::F16(lhs), GcuStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f16>(elem_count).w()?;
                let bias = dev.alloc::<f16>(n).w()?;
                let param = dev.get_gemm_launch_params(ubridge::DATATYPE::DataFp16, b, m, k, n);
                let kernel_name = "gemm_f16".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::GEMM)?;

                let cfg = GcuLaunchConfig::for_gemm();
                let params = (lhs, rhs, &out, &bias, 
                    param.input_dtype, b, m, k, n,
                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                    param.lhs_transpose, param.rhs_transpose,
                    param.alpha, param.beta, param.addmm_beta, param.bias,
                    param.sip_m, param.sip_k, param.sip_n
                );
                // println!("GEMM F16: [{} {}, {}, {}], SIP [{} {} {}]", b, m, k, n, param.sip_m, param.sip_k, param.sip_n);

                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F16(out)
            }
            (GcuStorageSlice::F32(lhs), GcuStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f32>(elem_count).w()?;
                let bias = dev.alloc::<f32>(n).w()?;
                let param = dev.get_gemm_launch_params(ubridge::DATATYPE::DataFp32, b, m, k, n);

                let kernel_name = "gemm_f32".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::GEMM)?;

                let cfg = GcuLaunchConfig::for_gemm();
                let params = (lhs, rhs, &out, &bias, 
                    param.input_dtype, b, m, k, n,
                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                    param.lhs_transpose, param.rhs_transpose,
                    param.alpha, param.beta, param.addmm_beta, param.bias,
                    param.sip_m, param.sip_k, param.sip_n
                );
                // println!("GEMM F32: [{} {}, {}, {}], SIP [{} {} {}]", b, m, k, n, param.sip_m, param.sip_k, param.sip_n);
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F32(out)
            }
            (GcuStorageSlice::F64(lhs), GcuStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let out = dev.alloc::<f64>(elem_count).w()?;
                let bias = dev.alloc::<f64>(n).w()?;
                let param = dev.get_gemm_launch_params(ubridge::DATATYPE::DataF64, b, m, k, n);
                let kernel_name = "gemm_f64".to_string();
                let func = dev.get_or_load_func(&kernel_name, ubridge::GEMM)?;

                let cfg = GcuLaunchConfig::for_gemm();
                let params = (lhs, rhs, &out, &bias, 
                    param.input_dtype, b, m, k, n,
                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                    param.lhs_transpose, param.rhs_transpose,
                    param.alpha, param.beta, param.addmm_beta, param.bias,
                    param.sip_m, param.sip_k, param.sip_n
                );
                unsafe { func.launch(cfg, params) }.w()?;
                GcuStorageSlice::F64(out)
            }
            _ => Err(GcuError::InternalError("dtype mismatch in matmul op"))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        // let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        let cfg = GcuLaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        // let ds = dev.htod_copy([dims, src_l.stride()].concat()).w()?;
        match (&self.slice, &mut dst.slice) {
            (GcuStorageSlice::BF16(src), GcuStorageSlice::BF16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_bf16", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F16(src), GcuStorageSlice::F16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    // println!("ucopy_f16 {}", el_count);
                    let func = dev.get_or_load_func("ucopy_f16", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F32(src), GcuStorageSlice::F32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_f32", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::U8(src), GcuStorageSlice::U8(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_u8", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::U32(src), GcuStorageSlice::U32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_u32", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::I64(src), GcuStorageSlice::I64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_i64", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?
                }
            }
            (GcuStorageSlice::F64(src), GcuStorageSlice::F64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.dtod_copy(&src, &mut dst).w()?
                } else {
                    let func = dev.get_or_load_func("ucopy_64", ubridge::UNARY)?;
                    let params = (el_count, &src, &dst);
                    unsafe { func.launch(cfg, params) }.w()?;
                }
            }
            _ => Err(GcuError::InternalError(
                "dtype mismatch in copy_strided op",
            ))?,
        }
        Ok(())
    }
}
