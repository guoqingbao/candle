use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, DType, Kernels, MetalKernelError, Source};
use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLResourceOptions, MTLSize};
use std::sync::Arc;

/// Temporary storage used by the multi-block Metal argsort. Keeping these
/// buffers in Candle's allocator lets repeated sorts reuse the same storage.
pub struct MlxSortScratch {
    values_0: Arc<Buffer>,
    values_1: Arc<Buffer>,
    indices_0: Arc<Buffer>,
    indices_1: Arc<Buffer>,
    partitions: Arc<Buffer>,
}

impl MlxSortScratch {
    pub fn new(
        values_0: Arc<Buffer>,
        values_1: Arc<Buffer>,
        indices_0: Arc<Buffer>,
        indices_1: Arc<Buffer>,
        partitions: Arc<Buffer>,
    ) -> Self {
        Self {
            values_0,
            values_1,
            indices_0,
            indices_1,
            partitions,
        }
    }

    fn allocate(device: &Device, dtype: DType, nrows: usize, ncols: usize, nblocks: usize) -> Self {
        let el_count = nrows * ncols;
        let bytes_len = (el_count * dtype.size_in_bytes()) as u64;
        Self {
            values_0: Arc::new(
                device.new_buffer(bytes_len, MTLResourceOptions::StorageModePrivate),
            ),
            values_1: Arc::new(
                device.new_buffer(bytes_len, MTLResourceOptions::StorageModePrivate),
            ),
            indices_0: Arc::new(
                device.new_buffer(el_count as u64 * 4, MTLResourceOptions::StorageModePrivate),
            ),
            indices_1: Arc::new(
                device.new_buffer(el_count as u64 * 4, MTLResourceOptions::StorageModePrivate),
            ),
            partitions: Arc::new(device.new_buffer(
                (nrows * (nblocks + 1)) as u64 * 4,
                MTLResourceOptions::StorageModePrivate,
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_arg_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    nrows: usize,
    ncols: usize,
    ncols_pad: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), crate::MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Sort, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&src, dst, ncols as i64, ncols_pad as i64));

    let thread_group_count = MTLSize {
        width: 1,
        height: nrows as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: ncols_pad as u64,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(src.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);
    encoder.set_threadgroup_memory_length(0, (ncols_pad * 4).max(16) as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

fn mlx_dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I64 => "int64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
    }
}

#[allow(clippy::too_many_arguments)]
pub fn multi_block_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    asc: bool,
    bn: usize,
    tn: usize,
    nblocks: usize,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
    scratch: &MlxSortScratch,
) -> Result<(), MetalKernelError> {
    let dtype_str = mlx_dtype_str(dtype);
    let dev_vals_0 = scratch.values_0.as_ref();
    let dev_vals_1 = scratch.values_1.as_ref();
    let dev_idxs_0 = scratch.indices_0.as_ref();
    let dev_idxs_1 = scratch.indices_1.as_ref();
    let block_partitions = scratch.partitions.as_ref();
    // Prepare command encoder
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    // Do blockwise sort
    {
        let direction = if asc { "" } else { "desc_" };
        let name = format!("sort_mbsort_{direction}{dtype_str}_uint32_bn{bn}_tn{tn}");
        let pipeline = kernels.load_pipeline(device, Source::MlxSort, name)?;
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(
            encoder,
            (
                &src,
                dev_vals_0,
                dev_idxs_0,
                /* size_sorted_axis */ ncols as i32,
                /* stride_sorted_axis */ 1i32,
                /* nc_dim */ 1i32,
                /* nc_shape */ nrows as i32,
                /* nc_str */ ncols as i32
            )
        );
        let thread_group_count = MTLSize {
            width: nblocks as u64,
            height: nrows as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: bn as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }
    // Do merges
    let mut ping = false;
    let mut merge_tiles = 2;
    let n_thr_per_group = usize::min(nblocks + 1, 1024);
    let direction = if asc { "" } else { "desc_" };
    let partition_name = format!("partition_mbsort_{direction}{dtype_str}_uint32_bn{bn}_tn{tn}");
    let merge_name = format!("merge_mbsort_{direction}{dtype_str}_uint32_bn{bn}_tn{tn}");
    while merge_tiles / 2 < nblocks {
        let (dev_vals_in, dev_vals_out) = if ping {
            (dev_vals_1, dev_vals_0)
        } else {
            (dev_vals_0, dev_vals_1)
        };
        let (dev_idxs_in, dev_idxs_out) = if ping {
            (dev_idxs_1, dev_idxs_0)
        } else {
            (dev_idxs_0, dev_idxs_1)
        };
        ping = !ping;
        // Do partition
        {
            let pipeline =
                kernels.load_pipeline(device, Source::MlxSort, partition_name.clone())?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    block_partitions,
                    dev_vals_in,
                    dev_idxs_in,
                    /* size_sorted_axis */ ncols as i32,
                    /* merge_tiles */ merge_tiles as i32,
                    /* n_blocks */ nblocks as i32
                )
            );
            let thread_group_count = MTLSize {
                width: 1,
                height: nrows as u64,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: n_thr_per_group as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        }
        // Do merge
        {
            let pipeline = kernels.load_pipeline(device, Source::MlxSort, merge_name.clone())?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    block_partitions,
                    dev_vals_in,
                    dev_idxs_in,
                    dev_vals_out,
                    dev_idxs_out,
                    /* size_sorted_axis */ ncols as i32,
                    /* merge_tiles */ merge_tiles as i32,
                    /* n_blocks */ nblocks as i32
                )
            );
            let thread_group_count = MTLSize {
                width: nblocks as u64,
                height: nrows as u64,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: bn as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        }
        merge_tiles *= 2;
    }
    let dev_idxs_out = if ping { dev_idxs_1 } else { dev_idxs_0 };
    // Copy output with appropriate strides
    let copy_kernel = match dtype {
        DType::U8 => crate::copy2d::U8,
        DType::U32 => crate::copy2d::U32,
        DType::I64 => crate::copy2d::I64,
        DType::BF16 => crate::copy2d::BFLOAT,
        DType::F16 => crate::copy2d::HALF,
        DType::F32 => crate::copy2d::FLOAT,
    };
    crate::call_copy2d(
        device,
        encoder,
        kernels,
        copy_kernel,
        dev_idxs_out,
        dst,
        /* d1 */ nrows,
        /* d2 */ ncols,
        /* src_s */ ncols,
        /* dst_s */ ncols,
        /* src_o_in_bytes */ 0,
        /*dst_o_in_bytes */ 0,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn block_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    asc: bool,
    bn: usize,
    tn: usize,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dtype_str = mlx_dtype_str(dtype);
    let direction = if asc { "" } else { "desc_" };
    let name = format!("carg_block_sort_{direction}{dtype_str}_uint32_bn{bn}_tn{tn}");
    let pipeline = kernels.load_pipeline(device, Source::MlxSort, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            &src,
            dst,
            ncols as i32,
            1i32,
            1i32,
            ncols as i32,
            ncols as i32
        )
    );
    let thread_group_count = MTLSize {
        width: 1,
        height: nrows as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: bn as u64,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(src.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_arg_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    asc: bool,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
    scratch: Option<&MlxSortScratch>,
) -> Result<(), MetalKernelError> {
    let tn = 8;
    let bn = match ncols.div_ceil(tn) {
        257.. if dtype.size_in_bytes() <= 4 => 512,
        129.. => 256,
        0..129 => 128,
    };
    let n_per_block = bn * tn;
    let n_blocks = ncols.div_ceil(n_per_block);
    if n_blocks > 1 {
        let owned_scratch;
        let scratch = match scratch {
            Some(scratch) => scratch,
            None => {
                owned_scratch = MlxSortScratch::allocate(device, dtype, nrows, ncols, n_blocks);
                &owned_scratch
            }
        };
        multi_block_sort(
            device, ep, kernels, dtype, asc, bn, tn, n_blocks, nrows, ncols, src, dst, scratch,
        )?
    } else {
        block_sort(
            device, ep, kernels, dtype, asc, bn, tn, nrows, ncols, src, dst,
        )?
    }
    Ok(())
}
