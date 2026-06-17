use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::loss;
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

use qmoe_engine::loader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    config: Option<PathBuf>,

    #[arg(short, long)]
    tokenizer: String,

    #[arg(short, long)]
    dataset: PathBuf,

    #[arg(long, default_value_t = 128)]
    block_size: usize,

    #[arg(long, default_value_t = 64)]
    stride: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    eprintln!("Loading tokenizer from {}...", args.tokenizer);
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let device = Device::Cpu;

    eprintln!("Loading model from {:?}...", args.model);
    let start_load = Instant::now();
    let (model, _config) = loader::load_model_from_safetensors(
        &args.model,
        args.config.clone(),
        &device,
    )?;
    let load_elapsed = start_load.elapsed();
    eprintln!("Model loaded in {:?}", load_elapsed);

    eprintln!("Reading dataset from {:?}...", args.dataset);
    let text = fs::read_to_string(&args.dataset)?;

    eprintln!("Tokenizing...");
    let start_tokenize = Instant::now();
    let encoding = tokenizer.encode(text, true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let tokenize_elapsed = start_tokenize.elapsed();
    eprintln!("Tokenized {} tokens in {:?}", tokens.len(), tokenize_elapsed);

    let total_tokens = tokens.len();
    if total_tokens <= args.block_size {
        anyhow::bail!(
            "Dataset has {} tokens, but block_size is {}. Need at least block_size + 1 tokens.",
            total_tokens, args.block_size
        );
    }

    let seq_len = args.block_size - 1;

    eprintln!(
        "Evaluating: {} tokens, block_size={}, stride={}, windows={}",
        total_tokens,
        args.block_size,
        args.stride,
        (total_tokens - args.block_size) / args.stride + 1
    );

    let start_eval = Instant::now();
    let mut total_loss = 0.0f64;
    let mut total_count: usize = 0;
    let mut chunk_idx: usize = 0;

    for start in (0..total_tokens - args.block_size + 1).step_by(args.stride) {
        let chunk = &tokens[start..start + args.block_size];
        let target = &chunk[1..]; // all but first token

        let input_ids: Vec<i64> = chunk[..seq_len].iter().map(|&t| t as i64).collect();
        let target_ids: Vec<i64> = target.iter().map(|&t| t as i64).collect();

        let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.reshape((1, seq_len))?;
        let target_tensor = Tensor::new(target_ids.as_slice(), &device)?;

        let (logits, _caches) = model.forward_prefill_all_logits(&input_tensor)?;
        // logits: (1, seq_len, vocab_size)
        let logits_2d = logits.squeeze(0)?; // (seq_len, vocab_size)

        // cross_entropy expects targets as u32
        let target_u32 = target_tensor.to_dtype(candle_core::DType::U32)?;
        let chunk_loss = loss::cross_entropy(&logits_2d, &target_u32)?;
        let loss_val = chunk_loss.to_scalar::<f32>()?;

        total_loss += loss_val as f64 * seq_len as f64;
        total_count += seq_len;
        chunk_idx += 1;

        if chunk_idx % 100 == 0 {
            eprintln!(
                "  chunk {}/{}: mean loss = {:.4}",
                chunk_idx,
                (total_tokens - args.block_size) / args.stride + 1,
                total_loss / total_count as f64
            );
        }

        // Free GPU/CPU memory
        drop(logits);
        drop(logits_2d);
        drop(input_tensor);
        drop(target_tensor);
        drop(target_u32);
        drop(chunk_loss);
    }

    let eval_elapsed = start_eval.elapsed();

    let avg_loss = total_loss / total_count as f64;
    let perplexity = avg_loss.exp();
    let tokens_per_sec = total_count as f64 / eval_elapsed.as_secs_f64();

    println!("──────────────────────────────────────────────");
    println!("Results:");
    println!("  Tokens evaluated: {}", total_count);
    println!("  Mean loss:        {:.6}", avg_loss);
    println!("  Perplexity:       {:.4}", perplexity);
    println!("  Wall time:        {:?}", eval_elapsed);
    println!("  Throughput:       {:.2} tokens/sec", tokens_per_sec);
    println!("──────────────────────────────────────────────");

    Ok(())
}
