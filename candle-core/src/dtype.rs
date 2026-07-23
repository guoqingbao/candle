//! Types for elements that can be stored and manipulated using tensors.
#![allow(clippy::redundant_closure_call)]
use crate::backend::BackendStorage;
use crate::{CpuStorage, CpuStorageRef, Error, Result};

/// The different types of elements allowed in tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    // Unsigned 8 bits integer.
    U8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 64 bits integer.
    I64,
    // Brain floating-point using half precision (16 bits).
    BF16,
    // Floating-point using half precision (16 bits).
    F16,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
    // FP8 E8M0: power-of-two exponent-only format (MX/OCP block scaling).
    // Stored as 1 byte per element. value = 2^(byte - 127).
    F8E8M0,
    // FP8 E4M3: 1 sign + 4 exponent (bias=7) + 3 mantissa bits.
    // Stored as 1 byte per element. max value ±448, no infinity.
    F8E4M3,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError(String);

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot parse '{}' as a dtype", self.0)
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "f8e8m0" => Ok(Self::F8E8M0),
            "f8e4m3" => Ok(Self::F8E4M3),
            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}

impl DType {
    /// String representation for dtypes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E8M0 => "f8e8m0",
            Self::F8E4M3 => "f8e4m3",
        }
    }

    /// The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::F8E8M0 => 1,
            Self::F8E4M3 => 1,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I64 => true,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::F8E8M0 | Self::F8E4M3 => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I64 => false,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::F8E8M0 | Self::F8E4M3 => true,
        }
    }

    /// Backend storage dtype used for byte-backed formats.
    ///
    /// `F8E4M3` / `F8E8M0` are stored as raw `U8` bytes on CUDA and Metal. Tensor
    /// ops that allocate temporary buffers must use this so copies stay dtype-compatible
    /// with safetensors-loaded FP8 weights (storage is U8, logical dtype is F8*).
    pub fn storage_dtype(self) -> Self {
        match self {
            Self::F8E8M0 | Self::F8E4M3 => Self::U8,
            other => other,
        }
    }
}

/// Decode an unsigned F8E8M0 / UE8M0 scale byte to f32: value = 2^(byte - 127).
///
/// Matches DeepSeek V4-Flash / MXFP8 block scales (`scale_fmt: "ue8m0"`):
/// - Uses `f32::from_bits(byte << 23)` for finite exponent-only values.
/// - Saturates byte 0xFF to 0xFD so the result stays finite (2^128 would overflow f32).
pub fn f8e8m0_decode(v: u8) -> f32 {
    let b = if v >= 0xFF { 0xFD } else { v };
    f32::from_bits((b as u32) << 23)
}

/// Decode an F8E4M3 byte to f32.
///
/// Format: 1 sign + 4 exponent (bias=7) + 3 mantissa bits.
/// Matches CUDA `__nv_cvt_fp8_to_halfraw(..., __NV_E4M3)`: bytes 0x7F and
/// 0xFF (all exponent + mantissa bits set) are NaN.
pub fn f8e4m3_decode(v: u8) -> f32 {
    let sign = (v >> 7) & 1;
    let exp = (v >> 3) & 0xF;
    let mant = v & 0x7;
    // NaN: exponent=0xF, mantissa=0x7 (hardware E4M3 / cuda_fp8).
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa)
        sign_f * (mant as f32) * (1.0f32 / 64.0) * (1.0f32 / 8.0)
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mantissa/8)
        let exp_val = 2.0f32.powi(exp as i32 - 7);
        sign_f * exp_val * (1.0 + mant as f32 / 8.0)
    }
}

pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
    + std::any::Any
    + crate::cpu::kernels::VecOps
{
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_>;
    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_> {
                CpuStorageRef::$dtype(data)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$dtype(data)
            }

            fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
    };
}
use half::{bf16, f16};

with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(i64, I64, |v: f64| v as i64, |v: i64| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);

pub trait IntDType: WithDType {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

impl IntDType for i64 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u8 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
