#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle_transformers::models::bigcode::{Config, GPTBigCode};
use clap::Parser;
use std::path::Path;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: GPTBigCode,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl TextGeneration {
    fn new(
        model: GPTBigCode,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize, batch_size: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        print!("{prompt}");
        std::io::stdout().flush()?;
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let mut start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let (context_size, past_len) = if self.model.config().use_cache && index > 0 {
                (1, tokens.len().saturating_sub(1))
            } else {
                (tokens.len(), 0)
            };
            if index == 1 {
                start_gen = std::time::Instant::now()
            }
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];

            let input = Tensor::new(ctxt, &self.device)?;
            let input = if batch_size > 1 {
                let dims = input.layout().dims();
                input
                    .broadcast_as((batch_size, if dims.len() > 1 { dims[1] } else { dims[0] }))?
                    .contiguous()?
            } else {
                input.unsqueeze(0)?
            };
            let logits = self.model.forward(&input, past_len)?;
            let logits = if batch_size > 1 {
                logits.narrow(0, 0, 1)?
            } else {
                logits
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        let throughput_per_req = (sample_len - 1) as f64 / dt.as_secs_f64();
        println!(
            "\n{} tokens generated ({} x {sample_len} tokens), throughput: {:.2} token/s ({} x {:.2} token/s)", sample_len * batch_size,
            batch_size, throughput_per_req * batch_size as f64, batch_size, throughput_per_req
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long, default_value = "bigcode/starcoderbase-1b")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    weight_path: Option<String>,

    #[arg(long, default_value_t = 1)]
    batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match &args.weight_path {
        Some(path) => Path::new(path).join("tokenizer.json"),
        None => repo.get("tokenizer.json")?,
    };

    let filenames = match &args.weight_path {
        Some(path) => vec![Path::new(path).join("model.safetensors")],
        None => vec!["model.safetensors"]
            .iter()
            .map(|f| repo.get(f))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::BF16, &device)? };
    let config = Config::starcoder_1b();
    let model = GPTBigCode::load(vb, config)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len, args.batch_size)?;
    Ok(())
}
