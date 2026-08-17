use super::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qx_quants, nearest_int,
};
use super::GgmlDType;
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16};
use rayon::prelude::*;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    type VecDotType: GgmlType;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;

    /// Generic implementation of the dot product without simd optimizations.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ2XXS {
    pub(crate) d: f16,
    pub(crate) qs: [u16; QK_K / 8],
}
const _: () = assert!(2 + QK_K / 8 * 2 == std::mem::size_of::<BlockIQ2XXS>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ2XS {
    pub(crate) d: f16,
    pub(crate) qs: [u16; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(2 + QK_K / 8 * 2 + QK_K / 32 == std::mem::size_of::<BlockIQ2XS>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ3XXS {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(2 + 3 * QK_K / 8 == std::mem::size_of::<BlockIQ3XXS>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4XS {
    pub(crate) d: f16,
    pub(crate) scales_h: u16,
    pub(crate) scales_l: [u8; QK_K / 64],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(2 + 2 + QK_K / 64 + QK_K / 2 == std::mem::size_of::<BlockIQ4XS>());

// The newer IQ formats are kept as native raw blocks. CUDA has dedicated
// dequantization/GEMM support for them; the CPU quantizer intentionally does
// not synthesize these formats.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ1S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u16; QK_K / 32],
}
const _: () = assert!(2 + QK_K / 8 + QK_K / 32 * 2 == std::mem::size_of::<BlockIQ1S>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ3S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) signs: [u8; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 64],
}
const _: () =
    assert!(2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64 == std::mem::size_of::<BlockIQ3S>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ2S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(2 + QK_K / 4 + QK_K / 32 + QK_K / 32 == std::mem::size_of::<BlockIQ2S>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4NL {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 16],
}
const _: () = assert!(2 + 16 == std::mem::size_of::<BlockIQ4NL>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ1M {
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u8; QK_K / 16],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(QK_K / 8 + QK_K / 16 + QK_K / 32 == std::mem::size_of::<BlockIQ1M>());

pub(crate) const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

pub(crate) const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub(crate) static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

pub(crate) static IQ2XS_GRID: [u64; 512] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b192b19,
    0x080808082b2b0808,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x0808081908192b2b,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x08080819192b0808,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b082b2b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908190819,
    0x0808191908191908,
    0x08081919082b0808,
    0x0808191919080819,
    0x0808191919081908,
    0x0808191919190808,
    0x08081919192b0819,
    0x080819192b080808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b08190808,
    0x0808192b082b192b,
    0x0808192b19080808,
    0x0808192b1908082b,
    0x0808192b2b081908,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808082b2b,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b0819192b08,
    0x08082b082b080808,
    0x08082b082b2b0808,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b1919080808,
    0x08082b192b080819,
    0x08082b192b082b19,
    0x08082b2b08080808,
    0x08082b2b082b0808,
    0x08082b2b082b2b08,
    0x08082b2b2b19192b,
    0x08082b2b2b2b0808,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x08190808192b0808,
    0x08190808192b2b2b,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b190808,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908190819,
    0x0819081908191908,
    0x08190819082b0808,
    0x0819081919080819,
    0x0819081919081908,
    0x0819081919190808,
    0x081908192b080808,
    0x081908192b191908,
    0x081908192b19192b,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b0808192b,
    0x0819082b08190808,
    0x0819082b19080808,
    0x0819082b192b0808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x08191908082b0808,
    0x0819190819080819,
    0x0819190819081908,
    0x0819190819082b19,
    0x0819190819190808,
    0x08191908192b1908,
    0x081919082b080808,
    0x0819191908080819,
    0x0819191908081908,
    0x0819191908190808,
    0x0819191919080808,
    0x0819192b08080808,
    0x0819192b08191908,
    0x0819192b19082b19,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b080819082b,
    0x08192b0819080808,
    0x08192b0819191908,
    0x08192b082b08192b,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b19192b192b,
    0x08192b2b19190819,
    0x08192b2b2b2b2b19,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808082b2b,
    0x082b080808190819,
    0x082b080808191908,
    0x082b0808082b0808,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08082b080808,
    0x082b08082b2b0808,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b081919080808,
    0x082b081919082b08,
    0x082b0819192b1919,
    0x082b082b08080808,
    0x082b082b082b082b,
    0x082b082b2b080808,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b1908082b2b19,
    0x082b190819080808,
    0x082b191908080808,
    0x082b191919080819,
    0x082b19191919082b,
    0x082b19192b192b19,
    0x082b192b08080819,
    0x082b192b08192b2b,
    0x082b192b2b2b192b,
    0x082b2b0808080808,
    0x082b2b0808082b08,
    0x082b2b0808082b2b,
    0x082b2b08082b0808,
    0x082b2b0819191919,
    0x082b2b082b082b08,
    0x082b2b082b2b082b,
    0x082b2b19192b2b08,
    0x082b2b192b190808,
    0x082b2b2b08082b08,
    0x082b2b2b082b0808,
    0x082b2b2b2b08082b,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x19080808192b0808,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x19080819082b0808,
    0x1908081919080819,
    0x1908081919081908,
    0x1908081919190808,
    0x190808192b080808,
    0x190808192b081919,
    0x190808192b2b082b,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b082b2b19,
    0x1908082b19080808,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808190819,
    0x1908190808191908,
    0x1908190808192b19,
    0x19081908082b0808,
    0x1908190819080819,
    0x1908190819081908,
    0x1908190819190808,
    0x190819082b080808,
    0x190819082b191908,
    0x1908191908080819,
    0x1908191908081908,
    0x1908191908190808,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819192b192b2b,
    0x1908192b08080808,
    0x1908192b08082b2b,
    0x1908192b19081908,
    0x1908192b19190808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b0819191908,
    0x19082b08192b082b,
    0x19082b1908080808,
    0x19082b1908190819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b19192b2b19,
    0x19082b2b08081908,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x19190808082b0808,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x1919080819190808,
    0x191908082b080808,
    0x1919081908080819,
    0x1919081908081908,
    0x1919081908190808,
    0x1919081908191919,
    0x1919081919080808,
    0x191908191908082b,
    0x1919082b08080808,
    0x1919082b19081908,
    0x1919082b2b2b2b2b,
    0x1919190808080819,
    0x1919190808081908,
    0x1919190808190808,
    0x19191908082b0819,
    0x1919190819080808,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b2b0819,
    0x1919191908080808,
    0x1919191908082b08,
    0x191919192b080808,
    0x191919192b082b08,
    0x1919192b082b0819,
    0x1919192b192b2b08,
    0x1919192b2b2b0819,
    0x19192b0808080808,
    0x19192b0808191908,
    0x19192b0819080819,
    0x19192b0819190808,
    0x19192b082b192b19,
    0x19192b1908192b2b,
    0x19192b1919080808,
    0x19192b191908082b,
    0x19192b2b2b081919,
    0x192b080808080819,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b080819191908,
    0x192b0808192b082b,
    0x192b08082b08192b,
    0x192b08082b2b2b19,
    0x192b081908080808,
    0x192b082b082b1908,
    0x192b082b19082b2b,
    0x192b082b2b19082b,
    0x192b190808080808,
    0x192b19080819192b,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919081919,
    0x192b19192b2b1908,
    0x192b2b0808080819,
    0x192b2b08192b2b2b,
    0x192b2b19082b1919,
    0x192b2b2b0808192b,
    0x192b2b2b19191908,
    0x192b2b2b192b082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b080808082b0808,
    0x2b080808082b2b2b,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808082b080808,
    0x2b0808082b08082b,
    0x2b0808082b2b2b08,
    0x2b0808082b2b2b2b,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b0808190808192b,
    0x2b08081908190808,
    0x2b08081919080808,
    0x2b08081919190819,
    0x2b08081919192b19,
    0x2b08082b08080808,
    0x2b08082b082b0808,
    0x2b08082b2b080808,
    0x2b08082b2b08082b,
    0x2b08082b2b2b0808,
    0x2b08082b2b2b2b08,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b0819082b082b19,
    0x2b08191908080808,
    0x2b08191919081908,
    0x2b0819192b2b1919,
    0x2b08192b08192b08,
    0x2b08192b192b2b2b,
    0x2b082b0808080808,
    0x2b082b0808082b08,
    0x2b082b08082b1919,
    0x2b082b0819192b2b,
    0x2b082b082b080808,
    0x2b082b082b08082b,
    0x2b082b082b2b2b08,
    0x2b082b190808192b,
    0x2b082b2b082b082b,
    0x2b082b2b2b080808,
    0x2b082b2b2b082b08,
    0x2b082b2b2b19192b,
    0x2b082b2b2b2b2b08,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b19080819080808,
    0x2b1908081919192b,
    0x2b1908082b081908,
    0x2b19081908080808,
    0x2b190819082b082b,
    0x2b190819192b1908,
    0x2b19082b1919192b,
    0x2b19082b2b082b19,
    0x2b19190808080808,
    0x2b19190808081919,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b19190819192b08,
    0x2b191919082b2b19,
    0x2b1919192b190808,
    0x2b1919192b19082b,
    0x2b19192b19080819,
    0x2b192b0819190819,
    0x2b192b082b2b192b,
    0x2b192b1919082b19,
    0x2b192b2b08191919,
    0x2b192b2b192b0808,
    0x2b2b080808080808,
    0x2b2b08080808082b,
    0x2b2b080808082b08,
    0x2b2b080808082b2b,
    0x2b2b0808082b0808,
    0x2b2b0808082b2b2b,
    0x2b2b08082b2b0808,
    0x2b2b081919190819,
    0x2b2b081919192b19,
    0x2b2b08192b2b192b,
    0x2b2b082b08080808,
    0x2b2b082b0808082b,
    0x2b2b082b08082b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b080808,
    0x2b2b082b2b2b0808,
    0x2b2b190819080808,
    0x2b2b19082b191919,
    0x2b2b192b192b1919,
    0x2b2b192b2b192b08,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808,
    0x2b2b2b08082b082b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08,
    0x2b2b2b1908081908,
    0x2b2b2b192b081908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b082b2b2b,
    0x2b2b2b2b2b190819,
    0x2b2b2b2b2b2b2b2b,
];

pub(crate) static IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

fn u16_slice_as_bytes(qs: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(qs.as_ptr().cast(), qs.len() * 2) }
}

fn iq_vec_dot_q8k<T: GgmlType<VecDotType = BlockQ8K>>(
    n: usize,
    xs: &[T],
    ys: &[BlockQ8K],
) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    if xs.len() < nb || ys.len() < nb {
        crate::bail!("vec_dot: block count mismatch");
    }
    let mut sumf = 0f32;
    let mut tmp = [0f32; QK_K];
    for i in 0..nb {
        T::to_float(&xs[i..i + 1], &mut tmp)?;
        let d8 = ys[i].d;
        for j in 0..QK_K {
            sumf += tmp[j] * ys[i].qs[j] as f32 * d8;
        }
    }
    Ok(sumf)
}

impl GgmlType for BlockIQ2XXS {
    const DTYPE: GgmlDType = GgmlDType::IQ2_XXS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("IQ2_XXS: {k} not divisible by {QK_K}");
        }
        if xs.len() != k / QK_K {
            crate::bail!("IQ2_XXS: block count mismatch");
        }
        for (block_idx, block) in xs.iter().enumerate() {
            let d = block.d.to_f32();
            let qs_bytes = u16_slice_as_bytes(&block.qs);
            for ib in 0..8usize {
                let q2 = &qs_bytes[8 * ib..8 * ib + 8];
                let aux32_g = LittleEndian::read_u32(&q2[0..4]);
                let aux32_s = LittleEndian::read_u32(&q2[4..8]);
                let aux8 = aux32_g.to_le_bytes();
                let block_scale = d * (0.5 + (aux32_s >> 28) as f32) * 0.25;
                for il in 0..4usize {
                    let grid = IQ2XXS_GRID[aux8[il] as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux32_s >> (7 * il)) & 127) as usize];
                    let base = block_idx * QK_K + 32 * ib + 8 * il;
                    for j in 0..8usize {
                        let sign = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        ys[base + j] = block_scale * grid[j] as f32 * sign;
                    }
                }
            }
        }
        Ok(())
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        crate::bail!("IQ quantization from float not supported")
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        iq_vec_dot_q8k(n, xs, ys)
    }
}

impl GgmlType for BlockIQ2XS {
    const DTYPE: GgmlDType = GgmlDType::IQ2_XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("IQ2_XS: {k} not divisible by {QK_K}");
        }
        if xs.len() != k / QK_K {
            crate::bail!("IQ2_XS: block count mismatch");
        }
        for (block_idx, block) in xs.iter().enumerate() {
            let d = block.d.to_f32();
            let qs_bytes = u16_slice_as_bytes(&block.qs);
            for ib in 0..8usize {
                let sc = block.scales[ib];
                let base = block_idx * QK_K + 32 * ib;
                for il in 0..4usize {
                    let q_val =
                        LittleEndian::read_u16(&qs_bytes[8 * ib + 2 * il..8 * ib + 2 * il + 2]);
                    let scale_nibble = ((sc >> (4 * (il / 2))) & 0x0f) as f32;
                    let block_scale = d * (0.5 + scale_nibble) * 0.25;
                    let grid_idx = (q_val & 0x1ff) as usize;
                    let sign_idx = ((q_val >> 9) & 0x7f) as usize;
                    let grid = IQ2XS_GRID[grid_idx].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[sign_idx];
                    for j in 0..8usize {
                        let sign = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        ys[base + 8 * il + j] = block_scale * grid[j] as f32 * sign;
                    }
                }
            }
        }
        Ok(())
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        crate::bail!("IQ quantization from float not supported")
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        iq_vec_dot_q8k(n, xs, ys)
    }
}

impl GgmlType for BlockIQ3XXS {
    const DTYPE: GgmlDType = GgmlDType::IQ3_XXS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("IQ3_XXS: {k} not divisible by {QK_K}");
        }
        if xs.len() != k / QK_K {
            crate::bail!("IQ3_XXS: block count mismatch");
        }
        for (block_idx, block) in xs.iter().enumerate() {
            let d = block.d.to_f32();
            let qs = &block.qs;
            for ib in 0..8usize {
                let q3 = &qs[8 * ib..8 * ib + 8];
                let gas0 = LittleEndian::read_u16(&qs[64 + 4 * ib..64 + 4 * ib + 2]) as u32;
                let gas1 = LittleEndian::read_u16(&qs[64 + 4 * ib + 2..64 + 4 * ib + 4]) as u32;
                let aux32 = gas0 | (gas1 << 16);
                let base_scale = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for il in 0..4usize {
                    let grid1 = IQ3XXS_GRID[q3[2 * il] as usize].to_le_bytes();
                    let grid2 = IQ3XXS_GRID[q3[2 * il + 1] as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * il)) & 127) as usize];
                    let base = block_idx * QK_K + 32 * ib + 8 * il;
                    for j in 0..4usize {
                        let s0 = if signs & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        ys[base + j] = base_scale * grid1[j] as f32 * s0;
                        ys[base + j + 4] = base_scale * grid2[j] as f32 * s1;
                    }
                }
            }
        }
        Ok(())
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        crate::bail!("IQ quantization from float not supported")
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        iq_vec_dot_q8k(n, xs, ys)
    }
}

impl GgmlType for BlockIQ4XS {
    const DTYPE: GgmlDType = GgmlDType::IQ4_XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("IQ4_XS: {k} not divisible by {QK_K}");
        }
        if xs.len() != k / QK_K {
            crate::bail!("IQ4_XS: block count mismatch");
        }
        for (block_idx, block) in xs.iter().enumerate() {
            let d = block.d.to_f32();
            let scales_h = block.scales_h;
            let scales_l = &block.scales_l;
            let qs = &block.qs;
            for ib in 0..8usize {
                let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
                let high = ((scales_h >> (2 * ib)) & 0x03) as u8;
                let scale = (low | (high << 4)) as i32 - 32;
                let block_scale = d * scale as f32;
                for il in 0..4usize {
                    let base = block_idx * QK_K + 32 * ib + 4 * il;
                    let q4 = &qs[16 * ib + 4 * il..16 * ib + 4 * il + 4];
                    for j in 0..4usize {
                        let lo = (q4[j] & 0x0f) as usize;
                        let hi = (q4[j] >> 4) as usize;
                        ys[base + j] = block_scale * KVALUES_IQ4NL[lo] as f32;
                        ys[base + 16 + j] = block_scale * KVALUES_IQ4NL[hi] as f32;
                    }
                }
            }
        }
        Ok(())
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
        crate::bail!("IQ quantization from float not supported")
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        iq_vec_dot_q8k(n, xs, ys)
    }
}

macro_rules! impl_native_iq_raw {
    ($block:ty, $dtype:expr, $blck:expr, $name:literal) => {
        impl GgmlType for $block {
            const DTYPE: GgmlDType = $dtype;
            const BLCK_SIZE: usize = $blck;
            type VecDotType = BlockQ8K;

            fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
                crate::bail!(concat!($name, " CPU dequantization is not implemented"))
            }

            fn from_float(_xs: &[f32], _ys: &mut [Self]) -> Result<()> {
                crate::bail!(concat!($name, " quantization from float is not supported"))
            }

            fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
                crate::bail!(concat!($name, " CPU dot product is not implemented"))
            }

            fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
                crate::bail!(concat!($name, " CPU dot product is not implemented"))
            }
        }
    };
}

impl_native_iq_raw!(BlockIQ1S, GgmlDType::IQ1_S, QK_K, "IQ1_S");
impl_native_iq_raw!(BlockIQ3S, GgmlDType::IQ3_S, QK_K, "IQ3_S");
impl_native_iq_raw!(BlockIQ2S, GgmlDType::IQ2_S, QK_K, "IQ2_S");
impl_native_iq_raw!(BlockIQ4NL, GgmlDType::IQ4_NL, 32, "IQ4_NL");
impl_native_iq_raw!(BlockIQ1M, GgmlDType::IQ1_M, QK_K, "IQ1_M");

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        if k % qk != 0 {
            crate::bail!("dequantize_row_q4_0: {k} is not divisible by {qk}")
        }

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        if k % qk != 0 {
            crate::bail!("{k} is not divisible by {}", qk);
        };
        let nb = k / qk;
        if ys.len() != nb {
            crate::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        let qk = QK8_0;
        if n % QK8_0 != 0 {
            crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
        }
        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0;
            for j in 0..qk / 2 {
                let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs.qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + qk / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        Ok(sumf)
    }
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        // ggml_vec_dot_q4_1_q8_1
        let qk = QK8_1;
        if n % qk != 0 {
            crate::bail!("vec_dot_q4_1_q8_1: {n} is not divisible by {qk}")
        }
        let nb = n / qk;
        if nb % 2 != 0 {
            crate::bail!("vec_dot_q4_1_q8_1: {n}, nb is not divisible by 2")
        }

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sumi = 0i32;

            for j in 0..qk / 2 {
                let v0 = xs.qs[j] as i32 & 0x0F;
                let v1 = xs.qs[j] as i32 >> 4;
                sumi += (v0 * ys.qs[j] as i32) + (v1 * ys.qs[j + qk / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        Ok(sumf)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q4_1
        let qk = Self::BLCK_SIZE;
        if ys.len() * qk != xs.len() {
            crate::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 4) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = u8::min(15, (x0 + 0.5) as u8);
                let xi1 = u8::min(15, (x1 + 0.5) as u8);

                *q = xi0 | (xi1 << 4);
            }
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK4_1 != 0 {
            crate::bail!("dequantize_row_q4_1: {k} is not divisible by {QK4_1}");
        }

        let nb = k / QK4_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();

            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;

                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        let qk = Self::BLCK_SIZE;
        if n % Self::BLCK_SIZE != 0 {
            crate::bail!("vec_dot_q5_0_q8_0: {n} is not divisible by {qk}")
        }
        let nb = n / qk;
        if nb % 2 != 0 {
            crate::bail!("vec_dot_q5_0_q8_0: {n}, nb is not divisible by 2")
        }
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(_n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = (((qh & (1u32 << j)) >> j) << 4) as u8;
                let xh_1 = ((qh & (1u32 << (j + 16))) >> (j + 12)) as u8;

                let x0 = ((xs.qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
                let x1 = ((xs.qs[j] >> 4) as i32 | xh_1 as i32) - 16;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        Ok(sumf)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q5_0
        let k = xs.len();
        if ys.len() * Self::BLCK_SIZE != k {
            crate::bail!("size mismatch {k} {} {}", ys.len(), Self::BLCK_SIZE)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            let mut amax = 0f32;
            let mut max = 0f32;
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -16.;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut qh = 0u32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let x0 = xs[j] * id;
                let x1 = xs[j + Self::BLCK_SIZE / 2] * id;
                let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
                let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
                ys.qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + Self::BLCK_SIZE / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh)
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK5_0 != 0 {
            crate::bail!("dequantize_row_q5_0: {k} is not divisible by {QK5_0}");
        }

        let nb = k / QK5_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        let qk = Self::BLCK_SIZE;
        if n % Self::BLCK_SIZE != 0 {
            crate::bail!("vec_dot_q5_1_q8_1: {n} is not divisible by {qk}")
        }
        let nb = n / qk;
        if nb % 2 != 0 {
            crate::bail!("vec_dot_q5_1_q8_1: {n}, nb is not divisible by 2")
        }

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;

                let x0 = (xs.qs[j] as i32 & 0xF) | xh_0 as i32;
                let x1 = (xs.qs[j] as i32 >> 4) | xh_1 as i32;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        Ok(sumf)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q5_1
        let qk = Self::BLCK_SIZE;
        if ys.len() * qk != xs.len() {
            crate::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 5) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            let mut qh = 0u32;
            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = (x0 + 0.5) as u8;
                let xi1 = (x1 + 0.5) as u8;

                *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + qk / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh);
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK5_1 != 0 {
            crate::bail!("dequantize_row_q5_1: {k} is not divisible by {QK5_1}");
        }

        let nb = k / QK5_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;

                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK8_0 != 0 {
            crate::bail!("dequantize_row_q8_0: {k} is not divisible by {QK8_0}");
        }

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q8_0
        let k = xs.len();
        if k % Self::BLCK_SIZE != 0 {
            crate::bail!("{k} is not divisible by {}", Self::BLCK_SIZE);
        };
        let nb = k / Self::BLCK_SIZE;
        if ys.len() != nb {
            crate::bail!(
                "size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
        Ok(())
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        let qk = QK8_0;
        if n % QK8_0 != 0 {
            crate::bail!("vec_dot_q8_0_q8_0: {n} is not divisible by {qk}")
        }

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        Ok(sumf)
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLCK_SIZE: usize = QK8_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> Result<f32> {
        unimplemented!("no support for vec-dot on Q8_1")
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        // quantize_row_q8_1
        let k = xs.len();
        if ys.len() * Self::BLCK_SIZE != k {
            crate::bail!("size mismatch {k} {} {}", ys.len(), Self::BLCK_SIZE)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut sum = 0i32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let v0 = xs[j] * id;
                let v1 = xs[j + Self::BLCK_SIZE / 2] * id;
                ys.qs[j] = f32::round(v0) as i8;
                ys.qs[j + Self::BLCK_SIZE / 2] = f32::round(v1) as i8;
                sum += ys.qs[j] as i32 + ys.qs[j + Self::BLCK_SIZE / 2] as i32;
            }
            ys.s = f16::from_f32(sum as f32) * ys.d;
        }
        Ok(())
    }

    fn to_float(_xs: &[Self], _ys: &mut [f32]) -> Result<()> {
        unimplemented!("no support for vec-dot on Q8_1")
    }
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q2k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q2k_q8k: {n} is not divisible by {QK_K}")
        }

        let mut sumf = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q2: &[_] = &x.qs;
            let mut q8: &[_] = &y.qs;
            let sc = &x.scales;

            let mut summs = 0;
            for (bsum, scale) in y.bsums.iter().zip(sc) {
                summs += *bsum as i32 * ((scale >> 4) as i32);
            }

            let dall = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let mut isum = 0;
            let mut is = 0;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0;
                for _ in 0..4 {
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    let mut isuml = 0;
                    for l in 0..16 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    isuml = 0;
                    for l in 16..32 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    shift += 2;
                    // adjust the indexing
                    q8 = &q8[32..];
                }
                // adjust the indexing
                q2 = &q2[32..];
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }

        Ok(sumf)
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L279
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        const Q4SCALE: f32 = 15.0;

        for (block, x) in group_for_quantization(xs, ys)? {
            //calculate scales and mins
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];

            for (j, x_scale_slice) in x.chunks(16).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, x_scale_slice);
            }
            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (j, scale) in scales.iter().enumerate().take(QK_K / 16) {
                    block.scales[j] = nearest_int(iscale * scale) as u8;
                }
                block.d = f16::from_f32(max_scale / Q4SCALE);
            } else {
                for j in 0..QK_K / 16 {
                    block.scales[j] = 0;
                }
                block.d = f16::from_f32(0.0);
            }

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (j, scale) in block.scales.iter_mut().enumerate() {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    *scale |= l << 4;
                }
                block.dmin = f16::from_f32(max_min / Q4SCALE);
            } else {
                block.dmin = f16::from_f32(0.0);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
        Ok(())
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();

            let mut is = 0;

            for (y_block, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                // Step by 32 over q.
                let mut shift = 0;
                let mut y_block_index = 0;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[..16] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[16..] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    shift += 2;
                }
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q3k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q3k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q3k_q8k: {n} is not divisible by {QK_K}")
        }

        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut auxs: [u32; 4] = [0; 4];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q3: &[u8] = &x.qs;
            let hmask: &[u8] = &x.hmask;
            let mut q8: &[i8] = &y.qs;

            aux32.fill(0);
            let mut a = &mut aux8[..];

            let mut m = 1;
            //Like the GGML original this is written this way to enable the compiler to vectorize it.
            for _ in 0..QK_K / 128 {
                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = (q3_val & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 2) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 4) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 6) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;
                q3 = &q3[32..];
            }

            a = &mut aux8[..];

            LittleEndian::read_u32_into(&x.scales, &mut auxs[0..3]);

            let tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            auxs[0] = (auxs[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            for aux in auxs {
                for scale in aux.to_le_bytes() {
                    let scale = i8::from_be_bytes([scale]);
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];

                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
        }

        Ok(sums.iter().sum())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
                scales[j] = make_q3_quants(x_scale_slice, 4, true);
            }

            // Get max scale by absolute value.
            let mut max_scale: f32 = 0.0;
            for &scale in scales.iter() {
                if scale.abs() > max_scale.abs() {
                    max_scale = scale;
                }
            }

            block.scales.fill(0);

            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for (j, scale) in scales.iter().enumerate() {
                    let l_val = nearest_int(iscale * scale);
                    let l_val = l_val.clamp(-32, 31) + 32;
                    if j < 8 {
                        block.scales[j] = (l_val & 0xF) as u8;
                    } else {
                        block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                    }
                    let l_val = l_val >> 4;
                    block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
                }
                block.d = f16::from_f32(1.0 / iscale);
            } else {
                block.d = f16::from_f32(0.0);
            }

            let mut l: [i8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }

        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        for (block, y) in group_for_dequantization(xs, ys)? {
            //Reconstruct the scales
            let mut aux = [0; 4];
            LittleEndian::read_u32_into(&block.scales, &mut aux[0..3]);

            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            //Transfer the scales into an i8 array
            let scales: &mut [i8] =
                unsafe { std::slice::from_raw_parts_mut(aux.as_mut_ptr() as *mut i8, 16) };

            let d_all = block.d.to_f32();
            let mut m = 1;
            let mut is = 0;

            // Dequantize both 128 long blocks
            // 32 qs values per 128 long block
            // Each 16 elements get a scale
            for (y, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                let mut shift = 0;
                for shift_scoped_y in y.chunks_exact_mut(32) {
                    for (scale_index, scale_scoped_y) in
                        shift_scoped_y.chunks_exact_mut(16).enumerate()
                    {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                            let new_y = dl
                                * (((qs[i + 16 * scale_index] >> shift) & 3) as i8
                                    - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                        4
                                    } else {
                                        0
                                    }) as f32;
                            *inner_y = new_y;
                        }
                        // 16 block finished => advance scale index
                        is += 1;
                    }
                    // 32 block finished => increase shift and m
                    shift += 2;
                    m <<= 1;
                }
            }
        }

        Ok(())
    }
}

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
        }

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q4 = &q4[..];
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q4[l] & 0xF) as i8;
                }
                a = &mut a[32..];
                for l in 0..32 {
                    a[l] = (q4[l] >> 4) as i8;
                }
                a = &mut a[32..];
                q4 = &q4[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        Ok(sumf + sums.iter().sum::<f32>())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
        Ok(())
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
        Ok(())
    }
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q5k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q5k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q5k_q8k: {n} is not divisible by {QK_K}")
        }

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q5 = &x.qs;
            let hm = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q5 = &q5[..];
            let mut m = 1u8;

            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q5[l] & 0xF) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                for l in 0..32 {
                    a[l] = (q5[l] >> 4) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                q5 = &q5[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        Ok(sumf + sums.iter().sum::<f32>())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L793
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        for (block, x) in group_for_quantization(xs, ys)? {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(31, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }
            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }

        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        for (block, y) in group_for_dequantization(xs, ys)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0;
            let mut u1 = 1;
            let mut u2 = 2;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let ql = &ql[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                    ys_index += 1;
                }
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                    ys_index += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q6k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q6k_q8k: {n} is not divisible by {QK_K}")
        }

        let mut aux8 = [0i8; QK_K];
        let mut aux16 = [0i16; 8];
        let mut sums = [0f32; 8];
        let mut aux32 = [0f32; 8];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let q4 = &x.ql;
            let qh = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0f32);

            for j in (0..QK_K).step_by(128) {
                let aux8 = &mut aux8[j..];
                let q4 = &q4[j / 2..];
                let qh = &qh[j / 4..];
                for l in 0..32 {
                    aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 32] =
                        (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 96] =
                        (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                }
            }

            for (j, &scale) in x.scales.iter().enumerate() {
                let scale = scale as f32;
                let q8 = &q8[16 * j..];
                let aux8 = &aux8[16 * j..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
                let q8 = &q8[8..];
                let aux8 = &aux8[8..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
            }

            let d = x.d.to_f32() * y.d;
            for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
                *sum += a * d;
            }
        }
        Ok(sums.iter().sum())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() * Self::BLCK_SIZE {
            crate::bail!(
                "quantize_row_q6k: size mismatch {} {} {}",
                xs.len(),
                ys.len(),
                Self::BLCK_SIZE
            )
        }
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in ys.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale = make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1);
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
        Ok(())
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize_row_q6k: {k} is not divisible by {QK_K}")
        }
        for (idx_x, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut ys[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
        Ok(())
    }
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        #[cfg(target_feature = "avx")]
        return super::avx::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        let qk = QK_K;
        if n % QK_K != 0 {
            crate::bail!("vec_dot_q8k_q8k: {n} is not divisible by {qk}")
        }

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * xs.d * ys.d
        }
        Ok(sumf)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            crate::bail!("quantize_row_q8k: {k} is not divisible by {QK_K}")
        }
        for (i, y) in ys.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &xs[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize_row_q8k: {k} is not divisible by {QK_K}")
        }
        for (i, x) in xs.iter().enumerate() {
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = x.d * q as f32
            }
        }
        Ok(())
    }
}

// https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L10605
pub fn matmul<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = k.div_ceil(T::BLCK_SIZE);
    let k_in_rhs_blocks = k.div_ceil(T::VecDotType::BLCK_SIZE);
    // TODO: Do not make this copy if the DotType is f32.
    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs, lhs_b)?
    }
    let lhs_b = lhs_b.as_slice();

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                T::vec_dot(k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();

        result?;
    }
    Ok(())
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = f16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = bf16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_bf16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = bf16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }
}
