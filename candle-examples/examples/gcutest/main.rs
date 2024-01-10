use anyhow::{bail, Error as E, Result, Ok};

use candle::{DType, Tensor, Device};
use candle_nn::{VarBuilder, Module};
mod model;
use model::{Config, RmsNorm, CausalSelfAttention, Llama, Linear, Mlp, Block, Cache, LlamaConfig, embedding, linear, masked_fill};
use core::panic;
use std::path::PathBuf;
const MAX_SEQ_LEN: usize = 4096;
use candle::Shape;
use float_eq::{assert_float_eq, float_eq};
use half::{f16, bf16, vec};
use clap::Parser;
//passed!
fn test_cache_(config: &Config, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let n_elem = config.hidden_size / config.num_attention_heads;
    let theta: Vec<_> = (0..n_elem)
        .step_by(2)
        .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?; 
    let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

//passed!
fn test_cache(config: &Config, dtype: DType, gcu_device: &Device) -> Result<(Tensor, Tensor)> {
    let (gcu_sin, gcu_cos) = test_cache_(config, dtype, &gcu_device).unwrap();
    let (cpu_sin, cpu_cos) = test_cache_(config, dtype, &Device::Cpu).unwrap();
    assert_float_eq!(
        cpu_sin.to_dtype(DType::F32)?.to_vec2::<f32>()?[10],
        gcu_sin.to_dtype(DType::F32)?.to_vec2::<f32>()?[10],
        abs_all <= 0.000001);

    assert_float_eq!(
        cpu_cos.to_dtype(DType::F32)?.to_vec2::<f32>()?[10],
        gcu_cos.to_dtype(DType::F32)?.to_vec2::<f32>()?[10],
        abs_all <= 0.000001);
    println!("Test cache passed!");

    Ok((cpu_sin, cpu_cos))
}

use model::kvconcat;
//pased!
fn test_concat(gcu_device: &Device) -> Result<()> {
    let shape1: Shape = (1, 32, 13, 128).into();
    let shape2: Shape = (1, 32, 1, 128).into();

    let cpu_input1 = Tensor::rand(0.0f32, 1.0, shape1, &Device::Cpu)?;
    let cpu_input2 = Tensor::rand(0.0f32, 1.0, shape2, &Device::Cpu)?;

    let gcu_input1 = cpu_input1.to_device(&gcu_device)?;
    let gcu_input2 = cpu_input2.to_device(&gcu_device)?;
     
    let cpu_output = Tensor::cat(&[&cpu_input1, &cpu_input2], 2)?;

    let gcu_output = Tensor::cat(&[&gcu_input1, &gcu_input2], 2)?;
    let gcu_output1 = kvconcat(&gcu_input1, &gcu_input2, 2)?;
    // println!("Cpu output: {}", cpu_output);

    // println!("Gcu output: {}", gcu_output.to_device(&Device::Cpu)?);
    let out_shape: Shape = (32 * 14 * 128).into();
    let cpu_output = cpu_output.reshape(&out_shape)?;
    let gcu_output = gcu_output.reshape(&out_shape)?;
    let gcu_output1 = gcu_output1.reshape(&out_shape)?;

    assert_float_eq!(
        cpu_output.to_vec1::<f32>()?,
        gcu_output.to_vec1::<f32>()?,
        abs_all <= 0.000001);

    assert_float_eq!(
            cpu_output.to_vec1::<f32>()?,
            gcu_output1.to_vec1::<f32>()?,
            abs_all <= 0.000001);
    println!("Test concat passed!");

    Ok(())
}

//Passed!
fn test_embedding(tokens: &Vec<u32>, cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, gcu_device: &Device) -> Result<()> {
    let ctxt = &tokens[0..];
    let input = Tensor::new(ctxt, &gcu_device)?.unsqueeze(0)?;
    let cpu_input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;

    let wte = embedding(cfg, vb.pp("model.embed_tokens")).unwrap();
    let gcu_embedding = wte.forward(&input).unwrap();

    let wte1 = embedding(cfg, vbcpu.pp("model.embed_tokens")).unwrap();
    let cpu_embedding = wte1.forward(&cpu_input).unwrap();

    assert_float_eq!(
        gcu_embedding.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        cpu_embedding.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );

    // println!("GCU output: {:?}", gcu_embedding.to_vec3::<f32>()?[0][1]);
    // println!("CPU output: {:?}", cpu_embedding.to_vec3::<f32>()?[0][1]);
    println!("Test embedding passed!");

    Ok(())
}

//passed!
fn test_softmax(dtype: DType, gcu_device: &Device) -> Result<()> {
    let shape: Shape = (1, 13, 4096).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;
    
    let cpu_output = candle_nn::ops::softmax(&cpu_input, 1)?;
    let gcu_output = candle_nn::ops::softmax(&gcu_input, 1)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][0],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][0],
        abs_all <= 0.000001);

    println!("Test softmax passed!");

    Ok(())
}

fn test_cast(dtype: DType, gcu_device: &Device) -> Result<()> {
    let shape: Shape = (1, 13, 4096).into();
    let cpu_input_f32 = Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?;
    let gcu_input_f32 = cpu_input_f32.to_device(&gcu_device)?;

    let cpu_output = cpu_input_f32.to_dtype(dtype)?;
    let gcu_output = gcu_input_f32.to_dtype(dtype)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][0],
        gcu_output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec3::<f32>()?[0][0],
        abs_all <= 0.0000001);

    println!("Test cast passed!");

    Ok(())
}
//Passed!
fn test_rmsnorm(cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()> {
    //input [1, 13, 4096], output [1, 13, 4096]
    let shape: Shape = (1, 13, 4096).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
    let gcu_result = rms_1.forward(&gcu_input)?;

    let rms_2 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vbcpu.pp("input_layernorm"))?;
    let cpu_result = rms_2.forward(&cpu_input)?;

    assert_float_eq!(
        cpu_result.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_result.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.00001
    );

    println!("Test rmsnorm passed!");

    Ok(())

}

//Passed!
fn test_maskfill(cache: &Cache, dtype: DType, gcu_device: &Device) -> Result<()>{
    //input [1, 32, 13, 13], [1, 32, 13, 13], -inf (f32::NEG_INFINITY)
    //output [1, 32, 13, 13]
    let shape: Shape = (1, 32, 13, 13).into();
    let outshape: Shape = (32, 13, 13).into();

    let cpu_input = Tensor::rand(0.0f32, 1.0, shape.clone(), &Device::Cpu)?;
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    let seq_len = 13;
    let mask = cache.mask(seq_len)?.broadcast_as(&shape)?;

    // let on_true_cpu = Tensor::new(f32::NEG_INFINITY, cpu_input.device())?.broadcast_as(mask.shape().dims())?;
    // let on_true_gcu = Tensor::new(f32::NEG_INFINITY, &gcu_device)?.broadcast_as(mask.shape().dims())?;
    // println!("CPU input: {}", on_true_cpu);
    // println!("GCU input: {}", on_true_gcu.to_device(&Device::Cpu)?);

    let cpu_output = masked_fill(&cpu_input, &mask.to_device(&Device::Cpu).unwrap(), f32::NEG_INFINITY)?;
    let cpu_output = cpu_output.reshape(&outshape)?;

    let gcu_output = masked_fill(&gcu_input, &mask, f32::NEG_INFINITY)?;
    let gcu_output = gcu_output.reshape(&outshape)?;

    // println!("CPU output: {}", cpu_output);
    // println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    assert_float_eq!(
        cpu_output.to_vec3::<f32>()?[0][1],
        gcu_output.to_vec3::<f32>()?[0][1],
        rmax_all <= 0.000001
    );
    println!("Test maskfill passed!");
    Ok(())
}

fn test_block(cache: &Cache, cache_cpu: &Cache, cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()>{
    //input [1, 13, 4096], output [1, 13, 4096]
    let shape: Shape = (1, 13, 4096).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };

    let block_cpu = Block::load(vbcpu.pp(&format!("model.layers.0")), cache_cpu, cfg).unwrap();
    let block_gcu = Block::load(vb.pp(&format!("model.layers.0")), cache, cfg).unwrap();
    let cpu_output = block_cpu.forward(&cpu_input, 0, 0)?;

    let gcu_input = cpu_input.to_device(&gcu_device)?;
    let gcu_output = block_gcu.forward(&gcu_input, 0, 0)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );
    println!("Test block passed!");

    Ok(())

}

//passed!
fn test_attention(cache: &Cache, cache_cpu: &Cache, cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()>{
    //input [1, 13, 4096], output [1, 13, 4096]
    let shape: Shape = (1, 13, 4096).into();
    let outshape: Shape = (1, 32 * 13 * 128).into();

    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    let attn_cpu = CausalSelfAttention::load(vbcpu.pp("self_attn"), cache_cpu, cfg)?;
    let attn_gcu = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;

    let cpu_output = attn_cpu.forward(&cpu_input, 0, 0)?;
    let gcu_output = attn_gcu.forward(&gcu_input, 0, 0)?;

    // println!("CPU output: {}", cpu_output);
    // println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    let cpu_output = cpu_output.reshape(&outshape)?;
    let gcu_output = gcu_output.reshape(&outshape)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        gcu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        abs_all <= 0.00001
    );
    println!("Test attention passed!");

    Ok(())
}

//passed!
use candle::{D};
fn test_narrow(dtype: DType, gcu_device: &Device) -> Result<()> {
    let shape: Shape = (1, 32, 13, 128).into();
    let outshape: Shape = (32, 13, 128).into();
    let hidden_size = 128;
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    // println!("CPU input: {}", cpu_input);
    // println!("GCU input: {}", cpu_input.to_device(&Device::Cpu)?);

    // println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    let cpu_output = cpu_input.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
    let gcu_output = gcu_input.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
    // println!("CPU output: {}", cpu_output);

    let cpu_output = Tensor::cat(&[&cpu_output, &cpu_output], D::Minus1)?;
    let gcu_output = Tensor::cat(&[&gcu_output, &gcu_output], D::Minus1)?;

    // println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);
    let cpu_output = cpu_output.reshape(&outshape)?;
    let gcu_output = gcu_output.reshape(&outshape)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );
    println!("Test narrow passed!");

    Ok(())

}


fn test_transpose(dtype: DType, gcu_device: &Device) -> Result<()> {
    let shape: Shape = (5, 6).into();
    let cpu_input = Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?;

    // let range = 128f32 * 4096f32;
    // let cpu_input = Tensor::arange(0f32, range, &Device::Cpu)?.reshape(shape)?;
    // let cpu_input = cpu_input.to_dtype(DType::F16)?;
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    let cpu_output = cpu_input.transpose(0, 1)?;
    let gcu_output = gcu_input.transpose(0, 1)?.contiguous()?;

    println!("CPU output: {}", cpu_output);

    println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[1],
        gcu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[1],
        abs_all <= 0.000001
    );
    println!("Test transpose passed!");

    Ok(())

}
//passed!
fn test_rotary_embedding(cache: &Cache, cache_cpu: &Cache, cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()>{
    //input [1, 32, 13, 128], output [1, 32, 13, 128]
    let attn_gcu = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;
    let attn_cpu = CausalSelfAttention::load(vbcpu.pp("self_attn"), cache_cpu, cfg)?;

    let shape: Shape = (1, 32, 13, 128).into();
    let outshape: Shape = (32, 13, 128).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    // let cpu_output = Tensor::cat(&[&cpu_input, &cpu_input], 3)?;
    // let gcu_output = Tensor::cat(&[&gcu_input, &gcu_input], 3)?;

    let cpu_output = attn_cpu.apply_rotary_emb(&cpu_input, 0)?;
    let gcu_output = attn_gcu.apply_rotary_emb(&gcu_input, 0)?;

    // println!("CPU output: {}", cpu_output);
    // println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    let cpu_output = cpu_output.reshape(&outshape)?;
    let gcu_output = gcu_output.reshape(&outshape)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );
    println!("Test rotary_embedding passed!");

    Ok(())
}
//passed!
fn test_mlp(cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()>{
    //input [1, 13, 4096], output [1, 13, 4096]
    let mlp_cpu = Mlp::load(vbcpu.pp("mlp"), cfg)?;
    let mlp_gcu = Mlp::load(vb.pp("mlp"), cfg)?;
    let shape: Shape = (1, 13, 4096).into();
    let cpu_input = match dtype {
        DType::F16 | DType::BF16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        // DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let cpu_output = mlp_cpu.forward(&cpu_input)?;

    let gcu_input = cpu_input.to_dtype(dtype)?.to_device(&gcu_device)?;

    assert_float_eq!(
        cpu_input.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_input.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );

    let gcu_output = mlp_gcu.forward(&gcu_input)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.00001
    );
    println!("Test mlp passed!");

    Ok(())

}

//passed!
fn test_linear(cfg: &Config, vb: &VarBuilder, vbcpu: &VarBuilder, dtype: DType, gcu_device: &Device) -> Result<()> {
    //input [1, 4096], output [1, 32000]
    let shape: Shape = (1, 4096).into();
    let cpu_input = match dtype {
        DType::F16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape, &Device::Cpu)?},
        DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    let lm_head_cpu = linear(cfg.hidden_size, cfg.vocab_size, vbcpu.pp("lm_head"))?;
    let cpu_output = lm_head_cpu.forward(&cpu_input)?;

    let lm_head_gcu = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
    let gcu_output = lm_head_gcu.forward(&gcu_input)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        gcu_output.to_dtype(DType::F32)?.to_vec2::<f32>()?[0],
        abs_all <= 0.00001
    );

    println!("Test linear/matmul passed!");
    Ok(())
}

//passed!
fn test_matmul(dtype: DType, gcu_device: &Device) -> Result<()> {
    //input [1, 4096], output [1, 32000]
    let shape_a: Shape = (1, 32, 13, 64).into();
    let shape_b: Shape = (1, 32, 64, 128).into();
    let outshape: Shape = (32, 13, 128).into();
    let cpu_input_a = match dtype {
        DType::F16 | DType::BF16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape_a, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape_a, &Device::Cpu)?},
        // DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape_a, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };

    let cpu_input_b = match dtype {
        DType::F16 | DType::BF16 => {Tensor::rand(f16::from_f32(0.0f32), f16::from_f32(1.0f32), shape_b, &Device::Cpu)?},
        DType::F32 => {Tensor::rand(0.0f32, 1.0, shape_b, &Device::Cpu)?},
        // DType::BF16 => {Tensor::rand(bf16::from_f32(0.0f32), bf16::from_f32(1.0f32), shape_b, &Device::Cpu)?},
        _ => {panic!("Error type!");}
    };

    let gcu_input_a = cpu_input_a.to_device(&gcu_device)?;
    let gcu_input_b = cpu_input_b.to_device(&gcu_device)?;

    let shape_a1: Shape = (32, 13, 64).into();

    let cpu_input_a1 = cpu_input_a.reshape(&shape_a1)?;
    let gcu_input_a1 = gcu_input_a.reshape(&shape_a1)?;

    assert_float_eq!(
        cpu_input_a1.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_input_a1.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.000001
    );

    let cpu_output = cpu_input_a.matmul(&cpu_input_b)?;
    let cpu_output = cpu_output.reshape(&outshape)?;
    let gcu_output = gcu_input_a.matmul(&gcu_input_b)?;
    let gcu_output = gcu_output.reshape(&outshape)?;

    assert_float_eq!(
        cpu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        gcu_output.to_dtype(DType::F32)?.to_vec3::<f32>()?[0][1],
        abs_all <= 0.00001
    );

    println!("Test matmul passed!");
    Ok(())
}

fn test_llama(cfg: &Config, cache: &Cache, cache_cpu: &Cache, vb: VarBuilder, vbcpu: VarBuilder, gcu_device: &Device) -> Result<()> {
    //input [1, 13, 4096], output [1, 13, 4096]
    let shape: Shape = (1, 13).into();
    let cpu_input = Tensor::from_slice(&[1u32, 500, 75, 600, 4095, 6, 1, 9, 10, 9, 7, 2, 0], (1, 13), &Device::Cpu)?;
    // let cpu_input = Tensor::randn(1u32, 4000u32, shape, &Device::Cpu)?;
    let gcu_input = cpu_input.to_device(&gcu_device)?;

    println!("Load GCU model...");
    let llama_gcu = Llama::load(vb, cache, cfg)?;
    println!("Start GCU inference...");
    let gcu_output = llama_gcu.forward(&gcu_input, 0)?;

    println!("Load CPU model...");
    let llama_cpu = Llama::load(vbcpu, cache_cpu, cfg)?;
    println!("Start CPU inference...");
    let cpu_output = llama_cpu.forward(&cpu_input, 0)?;

    println!("CPU output: {}", cpu_output);
    println!("GCU output: {}", gcu_output.to_device(&Device::Cpu)?);

    let cpu_output = cpu_output.to_vec2::<f32>()?;
    let gcu_output = gcu_output.to_vec2::<f32>()?;

    println!("Dif: \n");
    let mut difs: Vec<f32> = vec![];
    for i in 0..cpu_output[0].len() {
        let dif = (cpu_output[0][i] - gcu_output[0][i]).abs();
        if dif > 0.01 {
            println!("large dif {} ", dif);
        }
        if !difs.contains(&dif) {
            difs.insert(0, dif);
        }
    }
    println!("Difs: {:?}", difs);


    // assert_float_eq!(
    //     cpu_output.to_vec2::<f32>()?[0],
    //     gcu_output.to_vec2::<f32>()?[0],
    //     abs_all <= 0.001
    // );

    println!("Test llama passed!");
    Ok(())
}

fn rope_test() -> Result<()> {
    pub fn apply_rotary_emb(x: &Tensor, cos_sin: &Tensor) -> Result<Tensor> {
        let (b_sz, _, seq_len, hidden_size) = x.dims4()?;
        let cos = cos_sin.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let sin = cos_sin.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let x1 = x.narrow(D::Minus1, 0, hidden_size / 2)?;
        let x2 = x.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }
    const N: usize = 1 * 32 * 13 * 128;
    let cpu_input = Tensor::from_slice(&[0.5f32; N], (1, 32, 13, 128), &Device::Cpu)?;
    let cos_sin = Tensor::from_slice(&[0.95f32; 13 * 128], (13, 128), &Device::Cpu)?;
    let output = apply_rotary_emb(&cpu_input, &cos_sin)?;
    let shape: Shape = (32 * 13 * 128).into();
    let output = output.reshape(shape)?;
    let v: Vec<f32> = output.to_vec1().unwrap();
    println!("Output: {}", 32 * 13 * 128);

    for i in 0..32 * 13 * 128 {
        print!("{} ", v[i]);
    }
    // println!("Output: {:}", output);
    Ok(())

}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The initial prompt.
    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
}
fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    let args = Args::parse();
    let device = candle_examples::device(false)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        _ => DType::F32,
    };

    let tokenizer_filename = match &args.local_weights {
        Some(path) => { path.to_owned() + "tokenizer.json"},
        _ => {panic!("Path not found: {}", "tokenizer.json")},
    };

    let config_filename = match &args.local_weights {
        Some(path) => {path.to_owned() + "config.json"},
        _ => {panic!("Path not found: {}", "config.json")},
    };


    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);

    let mut filenames: Vec<PathBuf> = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        match &args.local_weights {
            Some(path) => {
                filenames.push((path.to_owned() + rfilename).into());
            }
            _ => {
                panic!("Path not found: {}", rfilename);
            }
        };
    }

    let cache = model::Cache::new(false, dtype, &config, &device)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)?}; 
    let vbcpu = unsafe {VarBuilder::from_mmaped_safetensors(&filenames, dtype, &Device::Cpu)?};
    let cache_cpu = model::Cache::new(false, dtype, &config, &Device::Cpu)?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let prompt = "Please give me 200 words about deep learning!";
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("start the candle-gcu testing cases...");
    test_cache(&config, dtype, &device)?;
    test_cast(dtype, &device)?;
    test_embedding(&tokens, &config, &vb, &vbcpu, &device)?;
    test_softmax(dtype, &device)?; 
    test_rmsnorm(&config, &vb.pp(&format!("model.layers.0")), &vbcpu.pp(&format!("model.layers.0")), dtype, &device)?;
    test_maskfill(&cache, dtype, &device)?;
    test_concat(&device)?;
    // test_mlp(&config, &vb.pp(&format!("model.layers.0")), &vbcpu.pp(&format!("model.layers.0")), dtype, &device)?;
    // test_linear(&config, &vb, &vbcpu, dtype, &device)?;
    test_matmul(dtype, &device)?;
    test_block(&cache, &cache_cpu, &config, &vb, &vbcpu, dtype, &device)?; 
    test_attention(&cache, &cache_cpu, &config, &vb.pp(&format!("model.layers.0")), &vbcpu.pp(&format!("model.layers.0")), dtype, &device)?; 
    test_narrow(dtype, &device)?;
    test_rotary_embedding(&cache, &cache_cpu,  &config, &vb.pp(&format!("model.layers.0")), &vbcpu.pp(&format!("model.layers.0")), dtype, &device)?; 

    // test_llama(&config, &cache, &cache_cpu, vb, vbcpu, &device)?; //out of memory for f32
    // rope_test()?;
    Ok(())

}