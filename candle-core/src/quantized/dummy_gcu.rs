#![allow(unused)]
use super::GgmlDType;
use crate::{GcuDevice, GcuStorage, Error, Result};

pub struct QGcuStorage {
    dtype: GgmlDType,
    device: GcuDevice,
}

impl QGcuStorage {
    pub fn zeros(_: &GcuDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithGcuSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &GcuDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<GcuStorage> {
        Err(Error::NotCompiledWithGcuSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<GcuStorage> {
        Err(Error::NotCompiledWithGcuSupport)
    }

    pub fn quantize(&mut self, _src: &GcuStorage) -> Result<()> {
        Err(Error::NotCompiledWithGcuSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &GcuStorage,
        _layout: &crate::Layout,
    ) -> Result<(GcuStorage, crate::Shape)> {
        Err(Error::NotCompiledWithGcuSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &GcuDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithGcuSupport)
}
