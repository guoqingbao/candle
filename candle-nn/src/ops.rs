//! Tensor ops.
//!

#[cfg(feature = "gcu")]
use candle::GcuStorage;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

#[cfg(not(feature = "gcu"))]
pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

#[cfg(not(feature = "gcu"))]
struct Sigmoid;

#[cfg(not(feature = "gcu"))]
impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
        };
        use candle::cuda_backend::SlicePtrOrNull;
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let shape = layout.shape();
                let dims = shape.dims();
                let el_count = shape.elem_count();
                let cfg = LaunchConfig::for_num_elems(el_count as u32);
                let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("usigmoid"), kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<T>(el_count) }.w()?;

                let params = (el_count, dims.len(), &ds, src, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(out)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;
        let device = storage.device();
        let dtype = storage.dtype();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, "sigmoid")?;
        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("sigmoid");
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };

        match (el_count % 2, dtype, layout.is_contiguous()) {
            (0, DType::BF16 | DType::F16, true) => {
                use candle_metal_kernels::unary::contiguous_tiled;
                let kernel_name = match dtype {
                    DType::F16 => contiguous_tiled::sigmoid::HALF,
                    DType::F32 => contiguous_tiled::sigmoid::FLOAT,
                    DType::BF16 => contiguous_tiled::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!(
                            "Metal contiguous_tiled unary sigmoid {dtype:?} not implemented"
                        )
                    }
                };
                candle_metal_kernels::call_unary_contiguous_tiled(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    el_count,
                    src,
                    &buffer,
                )
                .map_err(MetalError::from)?;
            }
            (_, _, true) => {
                use candle_metal_kernels::unary::contiguous;
                let kernel_name = match dtype {
                    DType::F16 => contiguous::sigmoid::HALF,
                    DType::F32 => contiguous::sigmoid::FLOAT,
                    DType::BF16 => contiguous::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!("Metal contiguous unary sigmoid {dtype:?} not implemented")
                    }
                };
                candle_metal_kernels::call_unary_contiguous(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    el_count,
                    src,
                    &buffer,
                )
                .map_err(MetalError::from)?;
            }
            (_, _, false) => {
                use candle_metal_kernels::unary::strided;
                let kernel_name = match dtype {
                    DType::F16 => strided::sigmoid::HALF,
                    DType::F32 => strided::sigmoid::FLOAT,
                    DType::BF16 => strided::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!("Metal strided unary sigmoid {dtype:?} not implemented")
                    }
                };
                let dst = candle_metal_kernels::BufferOffset::zero_offset(&buffer);
                candle_metal_kernels::call_unary_strided(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    layout.dims(),
                    src,
                    layout.stride(),
                    dst,
                )
                .map_err(MetalError::from)?;
            }
        }

        let new_storage = candle::MetalStorage::new(buffer, device.clone(), el_count, dtype);
        Ok((new_storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

#[cfg(not(feature = "gcu"))]
pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el) }.w()?;
                let params = (&src, &dst, n_cols as i32);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = storage.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle::bail!("softmax-last-dim is not implemented for {dtype:?}"),
        };

        let n = layout.stride().len();
        if !(layout.is_contiguous() && layout.stride()[n - 1] == 1) {
            candle::bail!("Non contiguous softmax-last-dim is not implemented");
        }

        let last_dim = layout.dims()[layout.shape().rank() - 1];
        let elem_count = layout.shape().elem_count();
        let output = device.new_buffer(elem_count, storage.dtype(), "softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, storage.dtype());
        Ok((newstorage, layout.shape().clone()))
    }

    #[cfg(feature = "gcu")]
    fn gcu_fwd(
        &self,
        storage: &GcuStorage,
        layout: &Layout,
    ) -> Result<(candle::GcuStorage, Shape)> {
        use candle::gcu_backend::ubridge;
        use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
        use candle::gcu_backend::ubridge::gcu_launch::GcuLaunchAsync;
        use candle::gcu_backend::ubridge::gcu_slice::GcuSlice;
        use candle::gcu_backend::ubridge::prelude::DeviceSlice;
        use candle::gcu_backend::{kernel_name, Map1, WrapErr};
        use candle::{backend::BackendStorage, gcu_backend::DeviceCopy};
        use candle::{GcuDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceCopy + WithDType>(
                &self,
                src: &GcuSlice<T>,
                dev: &GcuDevice,
                layout: &Layout,
            ) -> Result<GcuSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                // println!("dims: {:?}", dims);

                let dim_m1 = dims[dims.len() - 1];
                let (batch, chunks, last_dim_size) = if dims.len() == 1 {
                    (1, 1, dim_m1)
                } else {
                    (dims[0], el / dims[0] / dim_m1, dim_m1)
                };
                // println!("n_rows: {}, n_cols: {}", n_rows, n_cols);
                let mut cfg = dev.launch_cfg.clone();
                cfg.set_shared_memory(src.num_bytes() as u32 + 512 * 1024);
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), ubridge::REDUCE)?;
                let dst = dev.alloc::<T>(el).w()?;
                let params = (
                    src.device_ptr(),
                    dst.device_ptr(),
                    batch as i32,
                    chunks as i32,
                    last_dim_size as i32,
                );
                unsafe { func.launch(&cfg, params) }.w()?;
                Ok(dst)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::gcu_backend::GcuStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[cfg(not(feature = "cuda"))]
pub fn rms_norm(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(candle::D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(candle::D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

#[cfg(feature = "cuda")]
pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(candle::D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let sum2 = src
                        .iter()
                        .map(|&v| {
                            let v = v.as_();
                            v * v
                        })
                        .sum::<f32>();
                    let m = (sum2 / dim_m1 as f32 + eps).sqrt();
                    let m = T::from_f32(m).unwrap_or_else(T::nan);
                    for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                        *d = *s / m * *alpha
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("rmsnorm"), kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el) }.w()?;
                let params = (
                    &src,
                    &dst,
                    &alpha,
                    n_cols as i32,
                    block_size as i32,
                    self.eps,
                );
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle::bail!("rmsnorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous rmsnorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "rmsnorm")?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

#[cfg(feature = "cuda")]
pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[cfg(not(feature = "gcu"))]
#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
}

#[cfg(not(feature = "gcu"))]
impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map3, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map3 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                beta: &CudaSlice<T>,
                beta_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let beta = match beta_layout.contiguous_offsets() {
                    None => candle::bail!("beta has to be contiguous"),
                    Some((o1, o2)) => beta.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("layernorm"), kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el) }.w()?;
                let params = (
                    &src,
                    &dst,
                    &alpha,
                    &beta,
                    n_cols as i32,
                    block_size as i32,
                    self.eps,
                );
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, &s3.slice, l3, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "layernorm")?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

#[cfg(not(feature = "gcu"))]
pub fn layer_norm(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

//gcu apply_rotary_emb_qkv supports both rope and partial rope
#[cfg(feature = "gcu")]
pub fn apply_rotary_emb_qkv(
    query: &Tensor,
    key: &Tensor,
    cos_sin: &Tensor,
    _: &Tensor,
    index_positions: &Vec<i32>,
    split_dim: usize,
    query_key_transposed: bool,
    gpt_neox: bool,
) -> Result<(Tensor, Tensor)> {
    pub fn fused_rope(
        query: &Tensor,
        key: &Tensor,
        cos_sin: &Tensor,
        cos_sin_stride: i32,
        index_positions: &Vec<i32>,
        batch: i32,
        num_tokens: i32,
        q_head_size: i32,
        k_head_size: i32,
        hidden_size: i32,
        split_dim: i32, //used for partial rope
        gpt_neox: i32,
    ) -> Result<Tensor> {
        use candle::gcu_backend::Rope;
        let op = Rope {
            cos_sin_stride,
            index_positions: index_positions.clone(),
            batch,
            num_tokens,
            q_head_size,
            k_head_size,
            hidden_size,
            split_dim,
            gpt_neox,
        };
        query.apply_op3(key, cos_sin, op)
    }
    let cos_sin_dims = cos_sin.shape().dims();
    let cos_sin_stride = cos_sin_dims[cos_sin_dims.len() - 1];
    if query_key_transposed {
        //(b_sz, num_heads, seq_len, hidden_size)
        let (b_sz, q_head_size, seq_len, hidden_size) = query.dims4()?;
        let (_, k_head_size, _, _) = key.dims4()?;
        let _ = fused_rope(
            query,
            key,
            cos_sin,
            cos_sin_stride as i32,
            index_positions,
            b_sz as i32,
            seq_len as i32,
            q_head_size as i32,
            k_head_size as i32,
            hidden_size as i32,
            split_dim as i32,
            if gpt_neox { 1 } else { 0 },
        )?;
        Ok((query.contiguous()?, key.contiguous()?))
    } else {
        //NOTE: gpt_neox not for ChatGLM, seq_len in dim1 not for ChatGLM
        //(b_sz, seq_len, num_heads, hidden_size)
        let (b_sz, seq_len, q_head_size, hidden_size) = query.dims4()?;
        let (_, _, k_head_size, _) = key.dims4()?;
        let _ = fused_rope(
            query,
            key,
            cos_sin,
            cos_sin_stride as i32,
            index_positions,
            b_sz as i32,
            seq_len as i32,
            q_head_size as i32,
            k_head_size as i32,
            hidden_size as i32,
            split_dim as i32,
            if gpt_neox { 1 } else { 0 },
        )?;
        Ok((query.clone(), key.clone()))
    }
}

#[cfg(not(feature = "gcu"))]
pub fn apply_rotary_emb_qkv(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    split_dim: usize,
    query_key_transposed: bool,
    gpt_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if !gpt_neox {
        panic!("Not supported non-gpt-neox in apply_rotary_emb_qkv!");
    }
    use candle::D;
    fn rotate_half(xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(D::Minus1)?;
        let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
        let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
        Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
    }
    let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin))?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin))?;
    Ok((q_embed, k_embed))
}

#[cfg(not(feature = "gcu"))]
pub fn partial_rotary_emb_qkv(
    query: &Tensor,
    key: &Tensor,
    cos_sin: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    split_dim: usize,
    query_key_transposed: bool,
) -> Result<(Tensor, Tensor)> {
    let (_b_size, _num_heads, _seq_len, _headdim) = query.dims4()?; //must be this type of inputs
    use candle::D;
    let (rot_ndims, pass_ndims) = (split_dim, _headdim - split_dim);
    let query_rot = query.narrow(D::Minus1, 0, rot_ndims)?;
    let query_pass = query.narrow(D::Minus1, rot_ndims, pass_ndims)?;
    let key_rot = key.narrow(D::Minus1, 0, rot_ndims)?;
    let key_pass = key.narrow(D::Minus1, rot_ndims, pass_ndims)?;
    let (query_rot, key_rot) = apply_rotary_emb_qkv(
        &query_rot,
        &key_rot,
        &cos_sin,
        &sin,
        index_pos,
        0,
        query_key_transposed,
        true,
    )?;
    let query_states = Tensor::cat(&[query_rot, query_pass], D::Minus1)?.contiguous()?;
    let key_states = Tensor::cat(&[key_rot, key_pass], D::Minus1)?.contiguous()?;
    Ok((query_states, key_states))
}

#[cfg(feature = "gcu")]
pub fn kvconcat(ltensor: &Tensor, rtensor: &Tensor, concat_dim: i32) -> Result<Tensor> {
    use candle::gcu_backend::KVConcat;
    let op = KVConcat { concat_dim };
    //inputs for kvconcat must be contiguous tensors
    if ltensor.is_contiguous() && rtensor.is_contiguous() {
        ltensor.apply_op2(rtensor, op)
    } else if ltensor.is_contiguous() {
        ltensor.apply_op2(&rtensor.contiguous()?, op)
    } else if rtensor.is_contiguous() {
        let ltensor = ltensor.contiguous()?;
        ltensor.apply_op2(rtensor, op)
    } else {
        let ltensor = ltensor.contiguous()?;
        let rtensor = rtensor.contiguous()?;
        ltensor.apply_op2(&rtensor, op)
    }
}

#[cfg(not(feature = "gcu"))]
pub fn kvconcat(ltensor: &Tensor, rtensor: &Tensor, concat_dim: i32) -> Result<Tensor> {
    Tensor::cat(&[ltensor, &rtensor], concat_dim as usize)?.contiguous()
}

#[cfg(feature = "gcu")]
pub fn silu(xs: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::Silu;
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[cfg(feature = "gcu")]
pub fn relu(xs: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::ReLU;
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[cfg(feature = "gcu")]
pub fn gelu(xs: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::GeLU;
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[cfg(feature = "gcu")]
pub fn tanh(xs: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::Tanh;
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[cfg(feature = "gcu")]
pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::Sigmoid;
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[cfg(feature = "gcu")]
pub fn elu(xs: &Tensor, alpha: f64) -> Result<Tensor> {
    use candle::gcu_backend::Activation;
    let op = Activation::Elu(alpha);
    if xs.is_contiguous() {
        xs.apply_op1(op)
    } else {
        xs.contiguous()?.apply_op1(op)
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl crate::Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[cfg(feature = "gcu")]
pub fn gptq_matmul(
    x: &Tensor,
    qweight: &Tensor,
    scale: &Tensor,
    qzeros: &Option<Tensor>,
    g_idx: &Option<Tensor>,
    workspace: &Option<Tensor>,
    bits: i32,
    group_size: i32,
) -> Result<Tensor> {
    use candle::gcu_backend::GPTQMatMul;
    let op = GPTQMatMul {
        qzeros: qzeros.to_owned(),
        g_idx: g_idx.to_owned(),
        workspace: workspace.to_owned(),
        bits,
        group_size,
    };
    x.apply_op3(qweight, scale, op)
}

#[cfg(feature = "gcu")]
pub fn gptq_weight_repack(qweight: &Tensor) -> Result<Tensor> {
    use candle::gcu_backend::GPTQRepack;
    let op = GPTQRepack { bits: 4 };
    qweight.apply_op1(op)
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();

        let output = device.new_buffer(elem_count, q.dtype(), "sdpa_o")?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim =
            q_head == 32 || q_head == 64 || q_head == 96 || q_head == 128 || q_head == 256;

        const SDPA_FULL_THRESHOLD: usize = 2;

        let supports_sdpa_full =
            q_seq >= SDPA_FULL_THRESHOLD && supported_head_dim && q_head == k_head;
        let supports_sdpa_vector = q_seq == 1 && supported_head_dim;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("unsupported sdpa type {other:?}"),
        };

        let command_buffer = q.device().command_buffer()?;
        if supports_sdpa_vector {
            command_buffer.set_label("vector_attention");
            candle_metal_kernels::call_sdpa_vector(
                q.device().device(),
                &command_buffer,
                q.device().kernels(),
                q_l.start_offset(),
                q_l.dims(),
                q.buffer(),
                k_l.start_offset(),
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset(),
                v_l.stride(),
                v.buffer(),
                &output,
                self.scale,
                self.softcapping,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else if supports_sdpa_full {
            if q_l.dim(2)? != k_l.dim(2)? {
                candle::bail!(
                    "query and key sequence length must be equal if using full metal sdpa"
                )
            }

            command_buffer.set_label("full_attention");
            candle_metal_kernels::call_sdpa_full(
                q.device().device(),
                &command_buffer,
                q.device().kernels(),
                q_l.start_offset(),
                q_l.dims(),
                q.buffer(),
                k_l.start_offset(),
                k.buffer(),
                v_l.start_offset(),
                v.buffer(),
                &output,
                self.scale,
                self.softcapping,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else {
            candle::bail!("must be vector or full sdpa kernel");
        }

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, Shape::from_dims(&out_dims)))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// **Supported head dims:** 32, 64, 96, 128, 256.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Use an alternate kernel
///     - Requires `seq` == `kv_seq`
///     - GQA is not supported (requires `qhead` == `kv_head`)
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, softcapping: f32) -> Result<Tensor> {
    q.apply_op3_no_bwd(k, v, &Sdpa { scale, softcapping })
}

use crate::var_builder::ShardedVarBuilder as VarBuilder;
use crate::Linear;
use candle::backend::BackendStorage;
#[cfg(feature = "nccl")]
pub use candle::cuda_backend::cudarc::nccl::safe::{Comm, ReduceOp};
#[cfg(feature = "eccl")]
pub use candle::gcu_backend::ubridge::eccl::{Comm, Id, ReduceOp};
use candle::CustomOp1;
use candle::Module;
pub use std::rc::Rc;

#[cfg(all(not(feature = "nccl"), not(feature = "eccl")))]
struct Comm {}

#[cfg(all(not(feature = "nccl"), not(feature = "eccl")))]
impl Comm {
    //dummy Comm
    fn rank(&self) -> i32 {
        0
    }
    fn world_size(&self) -> i32 {
        1
    }
}

pub struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

pub struct TensorParallelRowLinear {
    linear: Linear,
    all_reduce: AllReduce,
}

struct AllReduce {
    comm: Rc<Comm>,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("AllReduce is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DeviceSlice;
        use candle::cuda_backend::WrapErr;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }

    #[cfg(all(feature = "gcu", feature = "eccl"))]
    fn gcu_fwd(&self, s: &candle::GcuStorage, l: &Layout) -> Result<(candle::GcuStorage, Shape)> {
        use candle::gcu_backend::ubridge::device_ptr::DeviceSlice;
        use candle::gcu_backend::WrapErr;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_gcu_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = dev.alloc::<bf16>(elem_count).w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::GcuStorage::wrap_gcu_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_gcu_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = dev.alloc::<f16>(elem_count).w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::GcuStorage::wrap_gcu_slice(dst, dev)
            }
            DType::F32 => {
                let s = s.as_gcu_slice::<f32>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = dev.alloc::<f32>(elem_count).w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::GcuStorage::wrap_gcu_slice(dst, dev)
            }
            dtype => candle::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    pub fn new(linear: Linear, comm: Rc<Comm>) -> Self {
        let all_reduce = AllReduce { comm };
        Self { linear, all_reduce }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)?.apply_op1_no_bwd(&self.all_reduce)
    }
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> crate::var_builder::Shard {
    crate::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

impl TensorParallelColumnLinear {
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank as usize, size as usize))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    pub fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| {
                vb.pp(p)
                    .get_with_hints((), "weight", shard(0, rank as usize, size as usize))
            })
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?.contiguous()?;
        Ok(Self::new(Linear::new(weight, None)))
    }
}

impl TensorParallelRowLinear {
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank as usize, size as usize))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }
}

#[cfg(feature = "gcu")]
fn update_cache<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    use candle::gcu_backend::ubridge;
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::ubridge::gcu_launch::GcuLaunchAsync;
    use candle::gcu_backend::{kernel_name, Map1, WrapErr};
    use candle::Storage;
    let dev = key.device().as_gcu_device()?;
    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Gcu(k) => k,
        _ => candle::bail!("key must be a gcu tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Gcu(v) => v,
        _ => candle::bail!("value must be a gcu tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Gcu(kc) => kc,
        _ => candle::bail!("key_cache must be a gcu tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Gcu(vc) => vc,
        _ => candle::bail!("value_cache must be a gcu tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Gcu(s) => s,
        _ => candle::bail!("slot_mapping must be a gcu tensor"),
    };

    let k_rank = k_l.stride().len();
    let v_rank = v_l.stride().len();
    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if k_rank != 3 || v_rank != 3 {
        candle::bail!("paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})")
    }

    if kc_rank != 5 {
        candle::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
        )
    }

    let k = k.as_gcu_slice::<T>()?;
    let v = v.as_gcu_slice::<T>()?;
    let kc = kc.as_gcu_slice::<T>()?;
    let vc = vc.as_gcu_slice::<T>()?;
    let s = s.as_gcu_slice::<i32>()?;

    // Get cuda views for all tensors
    let k = k.slice(k_l.start_offset()..);
    let v = v.slice(v_l.start_offset()..);
    let kc = kc.slice(kc_l.start_offset()..);
    let vc = vc.slice(vc_l.start_offset()..);
    let s = s.slice(s_l.start_offset()..);

    let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        candle::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = k_l.stride()[0] as i32;
    let value_stride = v_l.stride()[0] as i32;
    let func = dev.get_or_load_func(&kernel_name::<T>("reshape_and_cache"), ubridge::CACHE)?;
    let params = (
        k.device_ptr(),
        v.device_ptr(),
        kc.device_ptr(),
        vc.device_ptr(),
        s.device_ptr(),
        num_tokens as i32,
        num_heads as i32,
        head_size as i32,
        num_blocks as i32,
        block_size as i32,
        x as i32,
        key_stride,
        value_stride,
    );
    unsafe { func.launch(&dev.launch_cfg, params) }.w()?;
    Ok(())
}

#[cfg(feature = "gcu")]
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    use half::{bf16, f16};
    match key.dtype() {
        DType::F16 => update_cache::<f16>(key, value, key_cache, value_cache, slot_mapping),
        DType::BF16 => update_cache::<bf16>(key, value, key_cache, value_cache, slot_mapping),
        DType::F32 => update_cache::<f32>(key, value, key_cache, value_cache, slot_mapping),
        dt => {
            candle::bail!("reshape_and_cache is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}

#[cfg(feature = "gcu")]
pub fn paged_attention<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    q: &candle::GcuStorage,
    q_l: &Layout,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
) -> Result<(GcuStorage, Shape)> {
    use candle::gcu_backend::ubridge;
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::ubridge::gcu_launch::GcuLaunchAsync;
    use candle::gcu_backend::{kernel_name, WrapErr};
    use candle::GcuStorage;
    use candle::Storage;
    let dev = q.device();
    let out_shape = q_l.shape().clone();

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Gcu(kc) => kc,
        _ => candle::bail!("key_cache must be a gcu tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Gcu(vc) => vc,
        _ => candle::bail!("value_cache must be a gcu tensor"),
    };

    let (bt, bt_l) = block_tables.storage_and_layout();
    let bt = match &*bt {
        Storage::Gcu(bt) => bt,
        _ => candle::bail!("block_tables must be a gcu tensor"),
    };

    let (cl, cl_l) = context_lens.storage_and_layout();
    let cl = match &*cl {
        Storage::Gcu(cl) => cl,
        _ => candle::bail!("context_lens must be a gcu tensor"),
    };

    let q_rank = q_l.stride().len();
    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if q_rank != 3 {
        candle::bail!(
            "paged-attention expects `q` tensor to be of rank 3 \
            (q: {q_l:?})"
        )
    }

    if kc_rank != 5 {
        candle::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
            (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
            (value_cache: {vc_l:?})"
        )
    }

    let q = q.as_gcu_slice::<T>()?;
    let kc = kc.as_gcu_slice::<T>()?;
    let vc = vc.as_gcu_slice::<T>()?;
    let cl = cl.as_gcu_slice::<u32>()?; // Should be i32!
    let bt = bt.as_gcu_slice::<u32>()?; // Should be i32!

    // Get cuda views for all tensors
    let q = q.slice(q_l.start_offset()..);
    let kc = kc.slice(kc_l.start_offset()..);
    let vc = vc.slice(vc_l.start_offset()..);
    let cl = cl.slice(cl_l.start_offset()..);
    let bt = bt.slice(bt_l.start_offset()..);

    let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
    if !(head_size == 64
        || head_size == 80
        || head_size == 96
        || head_size == 112
        || head_size == 128
        || head_size == 192
        || head_size == 256)
    {
        candle::bail!("`head_size` must be one of 64, 80, 96, 112, 128 or 256");
    }

    let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

    if num_seqs_bt != num_seqs {
        candle::bail!(
            "shape mismatch block_tables {:?}, expected {:?}",
            bt_l.shape(),
            (num_seqs, max_num_blocks_per_seq)
        )
    }

    let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if head_size_kc != head_size / x {
        candle::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_seqs) != cl_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch context_lens {:?}, expected {:?}",
            cl_l.shape(),
            (num_seqs)
        )
    }

    let q_stride = q_l.stride()[0];
    let kv_block_stride = kc_l.stride()[0];
    let kv_head_stride = kc_l.stride()[1];

    // let partition_size = 512;
    // let max_num_partitions = (self.max_context_len + partition_size - 1) / partition_size;
    // let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
    //     && partition_size % block_size == 0;

    let elem_count = out_shape.elem_count();
    let out = dev.alloc::<T>(elem_count).w()?;
    let func = dev.get_or_load_func(&kernel_name::<T>("paged_attention_v1"), ubridge::ATTENTION)?;
    let params = (
        out.device_ptr(),
        q.device_ptr(),
        kc.device_ptr(),
        vc.device_ptr(),
        num_kv_heads as i32,
        softmax_scale,
        bt.device_ptr(),
        cl.device_ptr(),
        block_size as i32,
        max_context_len as i32,
        num_seqs as i32,
        num_heads as i32,
        head_size as i32,
        max_num_blocks_per_seq as i32,
        q_stride as i32,
        kv_block_stride as i32,
        kv_head_stride as i32,
        num_blocks as i32,
        softcapping,
    );
    unsafe { func.launch(&dev.launch_cfg, params) }.w()?;

    let storage = GcuStorage::wrap_gcu_slice(out, dev.clone());
    Ok((storage, out_shape))
}

#[cfg(feature = "gcu")]
fn topk_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    input: &Tensor,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use candle::gcu_backend::ubridge;
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::ubridge::gcu_launch::{GcuLaunchAsync, GcuLaunchConfig};
    use candle::gcu_backend::{kernel_name, Map1, WrapErr};
    use candle::Storage;
    use candle::gcu_backend::ubridge::ffi::{topk_f32, topk_f16, topk_bf16};
    let dev = input.device().as_gcu_device()?;
    let (value, input_l) = input.storage_and_layout();
    let shape = input_l.shape();
    let el_count = shape.elem_count();
    let stream = dev.stream_inner().unwrap();
    let value = match &*value {
        Storage::Gcu(k) => k,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };

    let rank = input_l.dims().len();
    assert!(rank <= 3);
    let value = value.as_gcu_slice::<T>()?;
    let value = value.slice(input_l.start_offset()..);
    let output = input.copy()?;
    // let func = dev.get_or_load_func(&kernel_name::<T>("topk"), ubridge::TOPK)?;

    let dims = if rank == 3 {
        shape.dims().to_vec()
    } else if rank == 2 {
        [1usize, shape.dims()[0], shape.dims()[1]].to_vec()
    } else {
        [1usize, 1usize, shape.dims()[0]].to_vec()
    };

    let indices = Tensor::arange(0u32, dims[2] as u32, input.device())?;
    // let indices = if rank > 1 {
    //     indices.broadcast_as(shape)?.contiguous()?
    // } else {
    //     indices
    // };
    let (index, indices_l) = indices.storage_and_layout();
    let index = match &*index {
        Storage::Gcu(k) => k,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };
    let index = index.as_gcu_slice::<u32>()?;
    let index = index.slice(indices_l.start_offset()..);

    fn next_power_of_2(x: usize) -> usize {
        let mut n = 1;
        while n < x {
            n *= 2
        }
        n
    }

    fn align_up(a: usize, b: usize) -> usize {
        ((a + b - 1) / b) * b
    }
    const SHARED_SIZE: i32 = 1024 * 1024 * (64 - 4);

    let workspace_size = dims[0] * (std::mem::size_of::<T>() + 4) * 24 * next_power_of_2(k) * align_up(dims[1], 128);
    let workspace_size = align_up(workspace_size, 512);
    let workspace = dev.alloc::<i8>(workspace_size).w()?;
    let (out, output_l) = output.storage_and_layout();
    let out = match &*out {
        Storage::Gcu(k) => k,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };
    let out = out.as_gcu_slice::<T>()?;
    let out = out.slice(output_l.start_offset()..);
    match input.dtype() {
        DType::F16 => {
            unsafe {
                topk_f16(value.device_ptr() as *const core::ffi::c_void, out.device_ptr() as *mut core::ffi::c_void, index.device_ptr() as *mut u32, 
                workspace.device_ptr(), 
                dims[0] as i32, dims[1] as i32, dims[2] as i32,
                2 as i32,
                k as i32, stream as *mut core::ffi::c_void);
            }

        }
        DType::BF16 => {
            unsafe {
                topk_bf16(value.device_ptr() as *const core::ffi::c_void, out.device_ptr() as *mut core::ffi::c_void, index.device_ptr() as *mut u32, 
                workspace.device_ptr(), 
                dims[0] as i32, dims[1] as i32, dims[2] as i32,
                2 as i32,
                k as i32, stream as *mut core::ffi::c_void);
            }

        }
        DType::F32 => {
            unsafe {
                topk_f32(value.device_ptr() as *const core::ffi::c_void, out.device_ptr() as *mut core::ffi::c_void, index.device_ptr() as *mut u32, 
                workspace.device_ptr(), 
                dims[0] as i32, dims[1] as i32, dims[2] as i32,
                2 as i32,
                k as i32, stream as *mut core::ffi::c_void);
            }
        }
        _=> { panic!("not supported data type!")}
    }
    let values = output.narrow(D::Minus1, 0, k)?;
    let indices = indices.narrow(D::Minus1, 0, k)?;
    Ok((values, indices))
}

#[cfg(feature = "gcu")]
pub fn topk(
    input: &Tensor,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use half::{bf16, f16};
    match input.dtype() {
        DType::F16 => topk_func::<f16>(input, k),
        DType::BF16 => topk_func::<bf16>(input, k),
        DType::F32 => topk_func::<f32>(input, k),
        dt => {
            candle::bail!("topk is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}