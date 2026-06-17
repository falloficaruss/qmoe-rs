use anyhow::Result;
use clap::Parser;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use qmoe_engine::{loader, moe, model, tensor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the quantized model weights (.safetensors)
    #[arg(short, long)]
    model: Option<String>,

    /// Path to the HuggingFace tokenizer.json file
    #[arg(short, long)]
    tokenizer: String,

    /// The prompt to generate from
    #[arg(short, long, default_value = "fn quicksort(arr: &mut [i32]) {")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 10)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    tracing::info!("Starting QMoE Engine...");

    // --- Load tokenizer ---------------------------------------------------
    tracing::info!("Loading tokenizer from {}", args.tokenizer);
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let device = Device::Cpu;

    let (model, _config) = if let Some(ref model_path) = args.model {
        tracing::info!("Loading real model from {}...", model_path);
        loader::load_model_from_safetensors(model_path, None, &device)?
    } else {
        tracing::info!("No --model path provided. Using fallback mock model initialization...");

        let config = model::ModelConfig {
            vocab_size: 102400,
            hidden_size: 64,
            num_layers: 1,
            moe: moe::MoEConfig {
                num_experts: 4,
                top_k: 1,
                hidden_dim: 64,
                intermediate_dim: 128,
            },
            ..Default::default()
        };

        // 1. Create a dummy VarMap to bind standard weights
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // 2. Generate dummy packed experts.
        tracing::info!("Mocking sub-1-bit packed experts in workspace...");
        let mut all_layers_experts = Vec::new();
        for l in 0..config.num_layers {
            let mut layer_experts = Vec::new();
            for e in 0..config.moe.num_experts {
                let gate_bytes = vec![0b01010101; (config.moe.intermediate_dim * config.hidden_size) / 4];
                let up_bytes = vec![0b01010101; (config.moe.intermediate_dim * config.hidden_size) / 4];
                let down_bytes = vec![0b01010101; (config.hidden_size * config.moe.intermediate_dim) / 4];

                let scales = Tensor::ones((config.moe.intermediate_dim,), candle_core::DType::F32, &device)?;
                let down_scales = Tensor::ones((config.hidden_size,), candle_core::DType::F32, &device)?;

                let gate_path = std::env::temp_dir().join(format!("gate_l{}_e{}.bin", l, e));
                let up_path = std::env::temp_dir().join(format!("up_l{}_e{}.bin", l, e));
                let down_path = std::env::temp_dir().join(format!("down_l{}_e{}.bin", l, e));

                std::fs::write(&gate_path, &gate_bytes)?;
                std::fs::write(&up_path, &up_bytes)?;
                std::fs::write(&down_path, &down_bytes)?;

                let gate_proj = tensor::PackedQMoETensor::mmap_from_file(&gate_path, (config.moe.intermediate_dim, config.hidden_size), scales.clone())?;
                let up_proj = tensor::PackedQMoETensor::mmap_from_file(&up_path, (config.moe.intermediate_dim, config.hidden_size), scales)?;
                let down_proj = tensor::PackedQMoETensor::mmap_from_file(&down_path, (config.hidden_size, config.moe.intermediate_dim), down_scales)?;

                layer_experts.push(moe::PackedExpert { gate_proj, up_proj, down_proj });
            }
            all_layers_experts.push(layer_experts);
        }

        // 3. Initialize DeepSeek-Coder-V2 scaffold
        tracing::info!("Initializing DeepSeek-Coder-V2 scaffold...");
        let all_layers_shared: Vec<Option<model::SharedExpert>> = (0..config.num_layers).map(|_| None).collect();
        let model = model::DeepSeekCoderV2::new(vb, &config, all_layers_experts, all_layers_shared)?;
        (model, config)
    };

    // 4. Tokenize the prompt
    tracing::info!("Tokenizing prompt...");
    let encoding = tokenizer.encode(args.prompt, true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
    let mut tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let prompt_len = tokens.len();
    tracing::info!("Encoded {} prompt tokens", prompt_len);

    // 5. Autoregressive generation with KV caching
    tracing::info!("Starting autoregressive token generation...");

    let start_time = std::time::Instant::now();

    // 5a. Prefill – process the full prompt, capture KV caches
    let input_tensor = Tensor::new(tokens.as_slice(), &device)?.reshape((1, tokens.len()))?;
    let (mut logits, mut caches) = model.forward_prefill(&input_tensor)?;

    // 5b. Decode loop – generate max_tokens tokens one at a time
    for step in 0..args.max_tokens {
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let next_token = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(0);

        tokens.push(next_token);
        tracing::info!("Step {}: Generated token {}", step, next_token);

        // Prepare input for the next step (single token)
        let next_input = Tensor::new(&[next_token], &device)?.reshape((1, 1))?;
        logits = model.forward_next(&next_input, &mut caches)?;
    }

    let elapsed = start_time.elapsed();
    let tokens_per_sec = (args.max_tokens as f64) / elapsed.as_secs_f64();

    // 6. Decode the full sequence (prompt + generated) back to text
    let all_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let output_text = tokenizer.decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

    // 7. Print results
    tracing::info!("──────────────────────────────────────────────");
    tracing::info!("{}", output_text);
    tracing::info!("──────────────────────────────────────────────");
    tracing::info!("================ BENCHMARKING ================");
    tracing::info!("Prompt tokens: {}", prompt_len);
    tracing::info!("Generated:     {} tokens", args.max_tokens);
    tracing::info!("Elapsed Time:  {:?}", elapsed);
    tracing::info!("Speed:         {:.2} tokens/sec", tokens_per_sec);
    tracing::info!("==============================================");
    Ok(())
}
