#include "cuda_utils.cuh"
#include <cuda_fp8.h>
#include<stdint.h>

template <typename S, typename T>
__device__ void cast_(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S *inp,
    T *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = inp[i];
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = inp[strided_i];
        }
    }
}

template <typename S, typename T, typename I>
__device__ void cast_through(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S *inp,
    T *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = static_cast<T>(static_cast<I>(inp[i]));
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = static_cast<T>(static_cast<I>(inp[strided_i]));
        }
    }
}


#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_<SRC_TYPENAME, DST_TYPENAME>(numel, num_dims, info, inp, out); \
} \

#define CAST_THROUGH_OP(SRC_TYPENAME, DST_TYPENAME, INT_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_through<SRC_TYPENAME, DST_TYPENAME, INT_TYPENAME>(numel, num_dims, info, inp, out); \
} \

#if __CUDA_ARCH__ >= 800
CAST_OP(__nv_bfloat16, __nv_bfloat16, cast_bf16_bf16)

// Vectorized bf16->f32 cast: 8 bf16 elements per float4 load
extern "C" __global__ void cast_bf16_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const __nv_bfloat16 *inp, float *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        if (numel >= 8 && is_aligned_16(inp)) {
            const size_t vec_numel = numel / 8;
            const float4 *inp4 = reinterpret_cast<const float4*>(inp);
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 v = inp4[i];
                const __nv_bfloat16 *bp = reinterpret_cast<const __nv_bfloat16*>(&v);
                float *outp = out + i * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    outp[j] = __bfloat162float(bp[j]);
                }
            }
            const size_t tail_start = vec_numel * 8;
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __bfloat162float(inp[i]);
            }
        } else {
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __bfloat162float(inp[i]);
            }
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __bfloat162float(inp[strided_i]);
        }
    }
}

// Vectorized f32->bf16 cast: 4 f32 elements per float4 load
extern "C" __global__ void cast_f32_bf16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float *inp, __nv_bfloat16 *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        if (numel >= 4 && is_aligned_16(inp)) {
            const size_t vec_numel = numel / 4;
            const float4 *inp4 = reinterpret_cast<const float4*>(inp);
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 v = inp4[i];
                __nv_bfloat16 *outp = out + i * 4;
                outp[0] = __float2bfloat16_rn(v.x);
                outp[1] = __float2bfloat16_rn(v.y);
                outp[2] = __float2bfloat16_rn(v.z);
                outp[3] = __float2bfloat16_rn(v.w);
            }
            const size_t tail_start = vec_numel * 4;
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __float2bfloat16_rn(inp[i]);
            }
        } else {
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __float2bfloat16_rn(inp[i]);
            }
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2bfloat16_rn(inp[strided_i]);
        }
    }
}

CAST_OP(__nv_bfloat16, uint32_t, cast_bf16_u32)
CAST_OP(__nv_bfloat16, double,   cast_bf16_f64)
CAST_OP(uint8_t, __nv_bfloat16, cast_u8_bf16)
CAST_OP(uint32_t, __nv_bfloat16, cast_u32_bf16)
CAST_OP(double,   __nv_bfloat16, cast_f64_bf16)
CAST_THROUGH_OP(__nv_bfloat16, uint8_t, float, cast_bf16_u8)
CAST_THROUGH_OP(__nv_bfloat16, __half,   float, cast_bf16_f16)
CAST_THROUGH_OP(__half,   __nv_bfloat16, float, cast_f16_bf16)
#else
#include <cuda.h>
#if CUDA_VERSION >= 11000
CAST_OP(__nv_bfloat16, float,    cast_bf16_f32)
CAST_OP(float,    __nv_bfloat16, cast_f32_bf16)
CAST_THROUGH_OP(__nv_bfloat16, uint8_t, float, cast_bf16_u8)
CAST_THROUGH_OP(__nv_bfloat16, __half,  float, cast_bf16_f16)
CAST_THROUGH_OP(__nv_bfloat16, double,  float, cast_bf16_f64)
CAST_THROUGH_OP(__half,   __nv_bfloat16, float, cast_f16_bf16)
CAST_THROUGH_OP(double,   __nv_bfloat16, float, cast_f64_bf16)
CAST_THROUGH_OP(uint8_t,   __nv_bfloat16, float, cast_u8_bf16)
#endif
#endif

#if __CUDA_ARCH__ >= 530
CAST_OP(__half, __half, cast_f16_f16)

CAST_THROUGH_OP(__half, uint8_t,  float, cast_f16_u8)
CAST_OP(__half, uint32_t, cast_f16_u32)
CAST_OP(__half, double,   cast_f16_f64)
CAST_OP(uint8_t,  __half, cast_u8_f16 )
CAST_OP(uint32_t, __half, cast_u32_f16)
CAST_OP(double,   __half, cast_f64_f16)

// Vectorized f16->f32 cast: 8 half elements per float4 load
extern "C" __global__ void cast_f16_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const __half *inp, float *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        if (numel >= 8 && is_aligned_16(inp)) {
            const size_t vec_numel = numel / 8;
            const float4 *inp4 = reinterpret_cast<const float4*>(inp);
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 v = inp4[i];
                const __half *hp = reinterpret_cast<const __half*>(&v);
                float *outp = out + i * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    outp[j] = __half2float(hp[j]);
                }
            }
            const size_t tail_start = vec_numel * 8;
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __half2float(inp[i]);
            }
        } else {
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __half2float(inp[i]);
            }
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __half2float(inp[strided_i]);
        }
    }
}

// Vectorized f32->f16 cast: 4 f32 elements per float4 load
extern "C" __global__ void cast_f32_f16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float *inp, __half *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        if (numel >= 4 && is_aligned_16(inp)) {
            const size_t vec_numel = numel / 4;
            const float4 *inp4 = reinterpret_cast<const float4*>(inp);
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 v = inp4[i];
                __half *outp = out + i * 4;
                outp[0] = __float2half_rn(v.x);
                outp[1] = __float2half_rn(v.y);
                outp[2] = __float2half_rn(v.z);
                outp[3] = __float2half_rn(v.w);
            }
            const size_t tail_start = vec_numel * 4;
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __float2half_rn(inp[i]);
            }
        } else {
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __float2half_rn(inp[i]);
            }
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2half_rn(inp[strided_i]);
        }
    }
}
#endif

CAST_OP(uint32_t, uint32_t, cast_u32_u32)
CAST_OP(uint32_t, uint8_t,  cast_u32_u8 )
CAST_OP(uint32_t, int64_t,  cast_u32_i64 )
CAST_OP(uint32_t, float,    cast_u32_f32)
CAST_OP(uint32_t, double,   cast_u32_f64)

CAST_OP(uint8_t, uint32_t, cast_u8_u32)
CAST_OP(uint8_t, uint8_t,  cast_u8_u8 )
CAST_OP(uint8_t, int64_t,  cast_u8_i64 )
CAST_OP(uint8_t, float,    cast_u8_f32)
CAST_OP(uint8_t, double,   cast_u8_f64)

CAST_OP(int64_t, uint32_t, cast_i64_u32)
CAST_OP(int64_t, uint8_t,  cast_i64_u8 )
CAST_OP(int64_t, int64_t,  cast_i64_i64 )
CAST_OP(int64_t, float,    cast_i64_f32)
CAST_OP(int64_t, double,   cast_i64_f64)

CAST_OP(float, uint8_t,  cast_f32_u8 )
CAST_OP(float, uint32_t, cast_f32_u32)
CAST_OP(float, int64_t,  cast_f32_i64 )
CAST_OP(float, float,    cast_f32_f32)
CAST_OP(float, double,   cast_f32_f64)

CAST_OP(double, uint8_t,  cast_f64_u8 )
CAST_OP(double, uint32_t, cast_f64_u32)
CAST_OP(double, int64_t,  cast_f64_i64 )
CAST_OP(double, float,    cast_f64_f32)
CAST_OP(double, double,   cast_f64_f64)

// F8_E8M0 / UE8M0 (power-of-two exponent-only scale) to F32/BF16/F16
// Decode: value = 2^(byte - 127) = reinterpret(byte << 23) as float.
// Saturate 0xFF -> 0xFD so 2^128 does not overflow f32 (DeepSeek V4 / MXFP8).
__device__ __forceinline__ float e8m0_to_f32(uint8_t v) {
    uint8_t b = (v >= 0xFF) ? 0xFD : v;
    return __uint_as_float((unsigned int)b << 23);
}

// Hardware E4M3 decode via CUDA FP8 cast. Matches tensor-core / Cutlass
// semantics: bytes 0x7F / 0xFF become NaN (not the software E4M3FN max).
__device__ __forceinline__ float e4m3_to_f32(uint8_t v) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(v, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half *>(&hr));
}

extern "C" __global__ void cast_f8e8m0_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, float *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = e8m0_to_f32(inp[i]);
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = e8m0_to_f32(inp[strided_i]);
        }
    }
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void cast_f8e8m0_bf16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, __nv_bfloat16 *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __float2bfloat16_rn(e8m0_to_f32(inp[i]));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2bfloat16_rn(e8m0_to_f32(inp[strided_i]));
        }
    }
}

extern "C" __global__ void cast_f8e8m0_f16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, __half *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __float2half_rn(e8m0_to_f32(inp[i]));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2half_rn(e8m0_to_f32(inp[strided_i]));
        }
    }
}
#endif

// F8_E4M3 to F32/BF16/F16 using CUDA hardware FP8 cast.
extern "C" __global__ void cast_f8e4m3_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, float *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = e4m3_to_f32(inp[i]);
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = e4m3_to_f32(inp[strided_i]);
        }
    }
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void cast_f8e4m3_bf16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, __nv_bfloat16 *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __float2bfloat16_rn(e4m3_to_f32(inp[i]));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2bfloat16_rn(e4m3_to_f32(inp[strided_i]));
        }
    }
}

extern "C" __global__ void cast_f8e4m3_f16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const uint8_t *inp, __half *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __float2half_rn(e4m3_to_f32(inp[i]));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __float2half_rn(e4m3_to_f32(inp[strided_i]));
        }
    }
}
#endif
