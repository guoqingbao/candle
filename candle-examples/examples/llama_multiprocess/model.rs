use super::MAX_SEQ_LEN;
use candle::backend::BackendStorage;
use candle::{CpuStorage, CustomOp1, DType, Device, IndexOp, Layout, Result, Shape, Tensor, D};
use candle_nn::ops::{shard, Comm, TensorParallelColumnLinear, TensorParallelRowLinear};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, Linear, Module, RmsNorm};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
pub type Config = candle_transformers::models::llama::LlamaConfig;
use candle_transformers::models::llama::{Llama3RopeConfig, Llama3RopeType};
use std::f32::consts::PI;

#[derive(Clone)]
pub struct Cache {
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
    cos_sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?.to_dtype(DType::F32)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos_sin =
            Tensor::cat(&[&idx_theta.cos()?, &idx_theta.sin()?], D::Minus1)?.contiguous()?; //must be contiguous tensor;
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self {
            kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
            cos,
            sin,
            cos_sin,
            device: device.clone(),
        })
    }
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    Ok(Linear::new(weight, None))
}

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

struct CausalSelfAttention {
    qkv_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: Cache,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.shape().dims4()?;
        let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        index_pos: usize,
        block_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.shape().dims3()?;

        let qkv = self.qkv_proj.forward(x)?;
        let hidden_size = self.num_attention_heads * self.head_dim;

        let q = qkv.i((.., .., ..self.num_attention_heads * self.head_dim))?;
        let k = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim
                ..self.num_attention_heads * self.head_dim
                    + self.num_key_value_heads * self.head_dim,
        ))?;
        let v = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim + self.num_key_value_heads * self.head_dim..,
        ))?;

        let (q, k, mut v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v.contiguous()?)
        };

        let mut input_positions = Vec::<i32>::new();
        for _ in 0..b_sz {
            input_positions.push(index_pos as i32);
        }
        let (q, mut k) = candle_nn::apply_rotary_emb_qkv(
            &q,
            &k,
            if q.device().is_gcu() {
                &self.cache.cos_sin
            } else {
                &self.cache.cos
            },
            &self.cache.sin,
            &input_positions,
            0,
            true,
            true,
        )?;

        let mut cache = self.cache.kvs.lock().unwrap();
        if let Some((cache_k, cache_v)) = &cache[block_idx] {
            // k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
            // v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
            k = candle_nn::kvconcat(cache_k, &k, 2)?;
            v = candle_nn::kvconcat(cache_v, &v, 2)?;
            let k_seq_len = k.dims()[1];
            if k_seq_len > MAX_SEQ_LEN {
                k = k
                    .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                    .contiguous()?
            }
            let v_seq_len = v.dims()[1];
            if v_seq_len > 2 * MAX_SEQ_LEN {
                v = v
                    .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                    .contiguous()?
            }
        }
        cache[block_idx] = Some((k.clone(), v.clone()));

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn = match attention_mask {
                None => attn,
                Some(mask) => attn.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            attn_weights.matmul(&v)?
        };

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        candle_transformers::utils::repeat_kv(x, n_rep)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let qkv_proj = TensorParallelColumnLinear::load_multi(
            vb.clone(),
            &["q_proj", "k_proj", "v_proj"],
            comm.clone(),
        )?;
        let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: cfg.num_key_value_heads() / comm.world_size(),
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            cache: cache.clone(),
        })
    }
}

struct Mlp {
    c_fc1: TensorParallelColumnLinear,
    c_fc2: TensorParallelColumnLinear,
    c_proj: TensorParallelRowLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, _cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let c_fc1 = TensorParallelColumnLinear::load(vb.pp("gate_proj"), comm.clone())?;
        let c_fc2 = TensorParallelColumnLinear::load(vb.pp("up_proj"), comm.clone())?;
        let c_proj = TensorParallelRowLinear::load(vb.pp("down_proj"), comm)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
    Ok(RmsNorm::new(weight, eps))
}

impl Block {
    fn new(rms_1: RmsNorm, attn: CausalSelfAttention, rms_2: RmsNorm, mlp: Mlp) -> Self {
        Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        index_pos: usize,
        block_idx: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, attention_mask, index_pos, block_idx)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg, comm.clone())?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, comm)?;
        let input_layernorm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.hidden_size, 1e-5, vb.pp("post_attention_layernorm"))?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Llama {
    fn new(
        wte: Embedding,
        blocks: Vec<Block>,
        ln_f: RmsNorm,
        lm_head: Linear,
        device: &Device,
        dtype: DType,
    ) -> Self {
        Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            device: device.clone(),
            dtype,
        }
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_size, seq_len) = x.shape().dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, index_pos)?;
            Some(mask)
        };
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, attention_mask.as_ref(), index_pos, block_idx)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cache,
                    cfg,
                    comm.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(
            wte,
            blocks,
            norm,
            lm_head,
            vb.device(),
            vb.dtype(),
        ))
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }
}
