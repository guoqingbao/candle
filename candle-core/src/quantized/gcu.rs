#![allow(unused_variables)]
use super::gguf_file::Value;
use super::{GgmlDType, QStorage};
pub use crate::gcu_backend::GcuDType;
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, gcu_backend::WrapErr};
use crate::{GcuDevice, GcuStorage, Result};
use half::{bf16, f16};
pub use ubridge;
use ubridge::device_ptr::DeviceSlice;
use ubridge::gcu_launch::GcuLaunchAsync;
use ubridge::gcu_slice::{GcuSlice, GcuView};
use ubridge::prelude::DevicePtr;

#[derive(Clone, Debug)]
pub struct QGcuStorage {
    data: GcuSlice<u8>,
    dtype: GgmlDType,
    device: GcuDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_GCU_MMV_X: usize = 32;
pub const GGML_GCU_MMV_Y: usize = 1;
pub const GCU_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const GCU_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    (p + q - 1) / q
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &GcuView<f32>,
    dst: &mut GcuSlice<u8>,
    elem_count: usize,
    ky: usize,
    dev: &GcuDevice,
) -> Result<()> {
    let cfg = &dev.launch_cfg;
    let kx = elem_count;
    let kx_padded = pad(kx, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, GCU_DEQUANTIZE_BLOCK_SIZE);
    let func = dev.get_or_load_func("quantize_q8_1", ubridge::QUANTIZED)?;
    // let cfg = cudarc::driver::LaunchConfig {
    //     grid_dim: (num_blocks as u32, ky as u32, 1),
    //     block_dim: (GCU_DEQUANTIZE_BLOCK_SIZE as u32, 1, 1),
    //     shared_mem_bytes: 0,
    // };
    let params = (
        src.device_ptr(),
        dst.device_ptr(),
        kx as i32,
        kx_padded as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

fn dequantize<T: GcuDType + crate::WithDType>(
    data: &GcuSlice<u8>,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &GcuDevice,
) -> Result<GcuStorage> {
    use crate::gcu_backend::kernel_name;
    let cfg = &dev.launch_cfg;
    let name = match dtype {
        GgmlDType::Q4_0 => "dequantize_block_q4_0",
        GgmlDType::Q4_1 => "dequantize_block_q4_1",
        GgmlDType::Q5_0 => "dequantize_block_q5_0",
        GgmlDType::Q5_1 => "dequantize_block_q5_1",
        GgmlDType::Q8_0 => "dequantize_block_q8_0",
        GgmlDType::Q2K => "dequantize_block_q2_k",
        GgmlDType::Q3K => "dequantize_block_q3_k",
        GgmlDType::Q4K => "dequantize_block_q4_k",
        GgmlDType::Q5K => "dequantize_block_q5_k",
        GgmlDType::Q6K => "dequantize_block_q6_k",
        GgmlDType::Q8K => "dequantize_block_q8_k",
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(&kernel_name::<T>(name), ubridge::QUANTIZED)?;
    let dst = dev.alloc::<T>(elem_count).w()?;

    let params = (data.device_ptr(), dst.device_ptr(), elem_count);
    unsafe { func.launch(cfg, params).w()? };
    Ok(GcuStorage::wrap_gcu_slice(dst, dev.clone()))
}

fn quantize(src: &GcuStorage, dst: &GcuSlice<u8>, dtype: GgmlDType, dev: &GcuDevice) -> Result<()> {
    use crate::gcu_backend::{GcuError, GcuStorageSlice};
    let cfg = &dev.launch_cfg;
    let name = match dtype {
        GgmlDType::Q4_0 => "quantize_block_q4_0",
        GgmlDType::Q4_1 => "quantize_block_q4_1",
        GgmlDType::Q5_0 => "quantize_block_q5_0",
        GgmlDType::Q5_1 => "quantize_block_q5_1",
        GgmlDType::Q8_0 => "quantize_block_q8_0",
        GgmlDType::Q2K => "quantize_block_q2_k",
        GgmlDType::Q3K => "quantize_block_q3_k",
        GgmlDType::Q4K => "quantize_block_q4_k",
        GgmlDType::Q5K => "quantize_block_q5_k",
        GgmlDType::Q6K => "quantize_block_q6_k",
        GgmlDType::Q8K => "quantize_block_q8_k",
        _ => crate::bail!("unsupported dtype for quantize {dtype:?}"),
    };

    let (kernel_name, src_ptr, src_bytes) = match &src.slice {
        GcuStorageSlice::BF16(slice) => (
            format!("{}_bf16", name),
            slice.device_ptr(),
            slice.num_bytes(),
        ),
        GcuStorageSlice::F16(slice) => (
            format!("{}_f16", name),
            slice.device_ptr(),
            slice.num_bytes(),
        ),
        GcuStorageSlice::F32(slice) => (
            format!("{}_f32", name),
            slice.device_ptr(),
            slice.num_bytes(),
        ),
        _ => Err(GcuError::InternalError(
            "invalid source dtype for quantization",
        ))?,
    };

    let func = dev.get_or_load_func(&kernel_name, ubridge::QUANTIZED)?;

    let params = (src_ptr, dst.device_ptr(), src_bytes, dst.num_bytes());
    unsafe { func.launch(cfg, params).w()? };
    Ok(())
}

fn dequantize_mul_mat_vec(
    data: &GcuSlice<u8>,
    y: &GcuView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &GcuDevice,
) -> Result<GcuStorage> {
    let cfg = &dev.launch_cfg;
    let data_elems = data.len() / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, ubridge::QUANTIZED)?;
    let dst = dev.alloc::<f32>(nrows).w()?;
    let block_num_y = ceil_div(nrows, GGML_GCU_MMV_Y);
    // let cfg = cudarc::driver::LaunchConfig {
    //     grid_dim: (block_num_y as u32, 1, 1),
    //     block_dim: (WARP_SIZE as u32, GGML_GCU_MMV_Y as u32, 1),
    //     shared_mem_bytes: 0,
    // };

    let params = (data, y, &dst, ncols as i32, nrows as i32);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(GcuStorage::wrap_gcu_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &GcuSlice<u8>,
    y: &GcuView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &GcuDevice,
) -> Result<GcuStorage> {
    let cfg = &dev.launch_cfg;
    let data_elems = data.len() / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc::<u8>(y_size_in_bytes).w()?;
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, ubridge::QUANTIZED)?;
    let dst = dev.alloc::<f32>(nrows * b_size).w()?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32 + 1) / 2, 4),
        5..=8 => ((nrows as u32 + 1) / 2, 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    // let cfg = cudarc::driver::LaunchConfig {
    //     grid_dim: (nblocks, 1, 1),
    //     block_dim: (WARP_SIZE as u32, nwarps, 1),
    //     shared_mem_bytes: 0,
    // };

    let params = (
        data,
        &y_q8_1,
        &dst,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(GcuStorage::wrap_gcu_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &GcuSlice<u8>,
    y: &GcuView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &GcuDevice,
) -> Result<GcuStorage> {
    let cfg = &dev.launch_cfg;
    let data_elems = data.len() / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_rows * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc::<u8>(y_size_in_bytes).w()?;
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, ubridge::QUANTIZED)?;
    let dst = dev.alloc::<f32>(x_rows * y_cols).w()?;
    // let cfg = cudarc::driver::LaunchConfig {
    //     grid_dim: (
    //         ceil_div(x_rows, mmq_y) as u32,
    //         ceil_div(y_cols, mmq_x) as u32,
    //         1,
    //     ),
    //     block_dim: (WARP_SIZE as u32, 4, 1),
    //     shared_mem_bytes: 0,
    // };

    let params = (
        /* vx */ data,
        /* vy */ &y_q8_1,
        /* dst */ &dst,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(GcuStorage::wrap_gcu_slice(dst, dev.clone()))
}

impl QGcuStorage {
    pub fn zeros(device: &GcuDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let data = device.alloc_zeros::<u8>(size_in_bytes).w()?;
        Ok(QGcuStorage {
            data,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &GcuDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<GcuStorage> {
        return dequantize::<f32>(&self.data, self.dtype, elem_count, self.device());
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<GcuStorage> {
        return dequantize::<f16>(&self.data, self.dtype, elem_count, self.device());
    }

    pub fn dequantize_bf16(&self, elem_count: usize) -> Result<GcuStorage> {
        return dequantize::<bf16>(&self.data, self.dtype, elem_count, self.device());
    }

    pub fn quantize(&mut self, src: &GcuStorage) -> Result<()> {
        quantize(src, &self.data, self.dtype, &src.device)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &GcuStorage,
        layout: &crate::Layout,
    ) -> Result<(GcuStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }
}

impl QGcuStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &GcuStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(GcuStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_gcu_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &GcuStorage,
        layout: &crate::Layout,
    ) -> Result<(GcuStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_gcu_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &GcuDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let data = device.htod_sync_copy(data).w()?;
    Ok(QStorage::Gcu(QGcuStorage {
        data,
        device: device.clone(),
        dtype: T::DTYPE,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gcu_quantize_q8_1() -> Result<()> {
        let dev = GcuDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        quantize_q8_1(&y.slice(..), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn gcu_mmv_q8_1() -> Result<()> {
        let dev = GcuDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        let mut xs = QGcuStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&GcuStorage::wrap_gcu_slice(y.clone(), dev.clone()))?;
        let gcu_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.slice(..),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = gcu_storage.as_gcu_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let gcu_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.slice(..),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = gcu_storage.as_gcu_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn gcu_mm_q8_1() -> Result<()> {
        let dev = GcuDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        let mut xs = QGcuStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&GcuStorage::wrap_gcu_slice(y.clone(), dev.clone()))?;
        let gcu_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.slice(..),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = gcu_storage.as_gcu_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }
}
