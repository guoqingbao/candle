use candle::{CpuStorage, GcuStorage, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use crate::{layer_norm, LayerNorm, LayerNormConfig, VarBuilder};

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
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
    let xs = xs.chunk(2, candle::D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

#[cfg(not(feature = "gcu"))]
pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    (xs.neg()?.exp()? + 1.0)?.recip()
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
    let mask = (rand.ge(&drop_p)? * scale)?.to_dtype(xs.dtype())?;
    xs * mask
}

#[derive(Debug)]
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
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el) }.w()?;
                let params = (src, &dst, n_cols as i32);
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
        use candle::{backend::BackendStorage, DType};
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
        .unwrap();
        let newstorage = candle::MetalStorage::new(output, device.clone(), storage.dtype());
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
        use candle::gcu_backend::ubridge::gcu_slice::GcuSlice;
        use candle::gcu_backend::ubridge::gcu_launch::GcuLaunchAsync;
        use candle::gcu_backend::{kernel_name, Map1, WrapErr};
        use candle::{GcuDevice, WithDType};
        use candle::{backend::BackendStorage, gcu_backend::DeviceCopy};

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
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);
                // println!("n_rows: {}, n_cols: {}", n_rows, n_cols);
                let cfg = dev.launch_cfg;
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), ubridge::REDUCE)?;
                let dst = dev.alloc::<T>(el).w()?;
                let params = (src.device_ptr(), dst.device_ptr(), n_rows, n_cols);
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
pub fn apply_rotary_emb_qkv(query: &Tensor, key: &Tensor, cos_sin: &Tensor, _: &Tensor, 
        index_pos: usize, split_dim: usize, query_key_transposed: bool, gpt_neox: bool) -> Result<(Tensor, Tensor)> {
    pub fn fused_rope(
        query: &Tensor,
        key: &Tensor,
        cos_sin: &Tensor,
        num_tokens: i32,
        q_head_size: i32,
        k_head_size: i32,
        hidden_size: i32,
        split_dim: i32, //used for partial rope
        gpt_neox: i32,
    ) -> Result<Tensor> {
        use candle::gcu_backend::Rope;
        let op = Rope {
            num_tokens,
            q_head_size,
            k_head_size,
            hidden_size,
            split_dim,
            gpt_neox
        };
        query.apply_op3(key, cos_sin, op)
    }

    if query_key_transposed {
        //(b_sz, q_len, num_kv_heads, head_dim)
        let (_, q_head_size, seq_len, hidden_size) = query.dims4()?;
        let (_, k_head_size, _, _) = key.dims4()?;
        //cost_sin must be type of float32
        let cos_sin = cos_sin.narrow(0, index_pos, seq_len)?;
        let _ = fused_rope(&query, &key, &cos_sin.contiguous()?, seq_len as i32, q_head_size as i32, k_head_size as i32, hidden_size as i32, split_dim as i32, 1)?;
        Ok((query.contiguous()?, key.contiguous()?))
    } else { //NOTE: gpt_neox not for ChatGLM, seq_len in dim1 not for ChatGLM
        //(b_sz, q_len, num_kv_heads, head_dim)
        let (seq_len_0, seq_len, q_head_size, hidden_size) = query.dims4()?;
        let (_, _, k_head_size, _) = key.dims4()?;
        //cost_sin must be type of float32
        let cos_sin = cos_sin.narrow(0, index_pos, if gpt_neox {seq_len} else { seq_len_0 })?;
        let _ = fused_rope(&query, &key, &cos_sin.contiguous()?, if gpt_neox {seq_len} else { seq_len_0 } as i32, q_head_size as i32, k_head_size as i32, hidden_size as i32, split_dim as i32, if gpt_neox { 1 } else { 0 })?;
        Ok((query.to_owned(), key.to_owned()))
    }

}

#[cfg(not(feature = "gcu"))]
pub fn apply_rotary_emb_qkv(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    split_dim: usize, query_key_transposed: bool, gpt_neox: bool
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
pub fn partial_rotary_emb_qkv(query: &Tensor, key: &Tensor, cos_sin: &Tensor, sin: &Tensor, index_pos: usize, split_dim: usize, query_key_transposed: bool) -> Result<(Tensor, Tensor)> {
    let (_b_size, _num_heads, _seq_len, _headdim) = query.dims4()?; //must be this type of inputs
    use candle::D;
    let (rot_ndims, pass_ndims) = (split_dim, _headdim - split_dim);
    let query_rot = query.narrow(D::Minus1, 0, rot_ndims)?;
    let query_pass = query.narrow(D::Minus1, rot_ndims, pass_ndims)?;
    let key_rot = key.narrow(D::Minus1, 0, rot_ndims)?;
    let key_pass = key.narrow(D::Minus1, rot_ndims, pass_ndims)?;
    let (query_rot, key_rot) =
        apply_rotary_emb_qkv(&query_rot, &key_rot, &cos_sin, &sin, index_pos, 0, query_key_transposed, true)?;
    let query_states = Tensor::cat(&[query_rot, query_pass], D::Minus1)?.contiguous()?;
    let key_states = Tensor::cat(&[key_rot, key_pass], D::Minus1)?.contiguous()?;
    Ok((query_states, key_states))
}

#[cfg(feature = "gcu")]
pub fn kvconcat(
    ltensor: &Tensor,
    rtensor: &Tensor,
    concat_dim: i32,
) -> Result<Tensor> {
    use candle::gcu_backend::KVConcat;
    let op = KVConcat {
        concat_dim
    };
    //inputs for kvconcat must be contiguous tensors
    if ltensor.is_contiguous() && rtensor.is_contiguous() {
        ltensor.apply_op2(&rtensor, op)
    } else if ltensor.is_contiguous() {
        ltensor.apply_op2(&rtensor.contiguous()?, op)
    } else if rtensor.is_contiguous() {
        let ltensor = ltensor.contiguous()?;
        ltensor.apply_op2(&rtensor, op)
    } else {
        let ltensor = ltensor.contiguous()?;
        let rtensor = rtensor.contiguous()?;
        ltensor.apply_op2(&rtensor, op)
    }
}

#[cfg(not(feature = "gcu"))]
pub fn kvconcat(
    ltensor: &Tensor,
    rtensor: &Tensor,
    concat_dim: i32,
) -> Result<Tensor> {
    Tensor::cat(&[ltensor, &rtensor], concat_dim as usize)?.contiguous()
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct LayerRmsNorm(LayerNorm);

impl LayerRmsNorm {
    pub fn new(rms: bool, weight: Tensor, eps: f64) -> Self {
        if rms {
            Self(LayerNorm::rms_norm(weight, eps))
        } else {
            Self(LayerNorm::new_no_bias(weight, eps))
        }
    }

    pub fn into_inner(self) -> LayerNorm {
        self.0
    }
}

#[cfg(not(feature = "gcu"))]
impl crate::Module for LayerRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

#[cfg(feature = "gcu")]
impl crate::Module for LayerRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.device().is_gcu() {
            use candle::gcu_backend::LayerNorm;
            let op = LayerNorm {
                eps: self.0.eps as f32,
                remove_mean: self.0.remove_mean,
                affine: self.0.bias.is_some(),
            };
            let x = xs.contiguous()?;
            match &self.0.bias {
                Some(bias) => x.apply_op3(&self.0.weight, &bias, op),
                None => x.apply_op3(&self.0.weight, &self.0.weight, op),
            }
        } else {
            self.0.forward(xs)
        }
    }
}

pub fn rms_norm_fused(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<LayerRmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(LayerRmsNorm(layer_norm(size, config, vb)?))
}

pub fn rms_norm_fused_shifted(size: usize, eps: f64, vb: crate::VarBuilder, shift: f64) -> Result<LayerRmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    let mut ln = layer_norm(size, config, vb)?;
    ln.weight = (ln.weight + shift)?;
    Ok(LayerRmsNorm(ln))
}

pub fn layer_norm_fused<C: Into<LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerRmsNorm> {
    Ok(LayerRmsNorm(layer_norm(size, c, vb)?))
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
