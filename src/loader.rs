use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::path::Path;

use crate::model::{DeepSeekCoderV2, ModelConfig, SharedExpert};
use crate::moe::{MoEConfig, PackedExpert};
use crate::tensor::PackedQMoETensor;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct ConfigJson {
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_local_experts: Option<usize>,
    pub top_k: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub hidden_dim: Option<usize>,
    pub intermediate_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
}

/// Check whether a tensor key should be skipped by standard weight loading
/// (i.e., it's a packed expert weight or scale loaded manually).
fn is_expert_tensor(name: &str) -> bool {
    let is_weight_or_scale = name.ends_with(".weight") || name.ends_with(".scales");
    if !is_weight_or_scale {
        return false;
    }
    // Per-expert: layers.{i}.moe.experts.{e}.{proj}.weight/scales
    if name.contains(".moe.experts.") {
        return true;
    }
    // Shared expert: layers.{i}.moe.shared_experts.{proj}.weight/scales
    if name.contains(".moe.shared_experts.") {
        return true;
    }
    // Flattened expert: layers.{i}.moe.gate_proj.weight/scales
    //                or layers.{i}.moe.up_proj.weight/scales
    //                or layers.{i}.moe.down_proj.weight/scales
    if name.contains(".moe.gate_proj.") || name.contains(".moe.up_proj.") || name.contains(".moe.down_proj.") {
        return true;
    }
    false
}

fn load_standard_weights(
    safetensors: &SafeTensors<'_>,
    varmap: &mut VarMap,
    device: &Device,
) -> Result<()> {
    use candle_core::safetensors::Load;

    let mut varmap_data = varmap.data().lock().unwrap();

    for name in safetensors.names() {
        if is_expert_tensor(name) {
            continue;
        }
        let view = safetensors.tensor(name)?;
        let tensor = view.load(device)?;
        let tensor = tensor.to_dtype(DType::F32)?;
        let var = Var::from_tensor(&tensor)?;
        varmap_data.insert(name.to_string(), var);
    }

    Ok(())
}

/// Deduce MLA head dimensions from tensor shapes.
pub fn deduce_config(safetensors: &SafeTensors<'_>) -> Result<ModelConfig> {
    let mut config = ModelConfig::default();

    // Gate gating weight for num_experts + hidden_size
    for name in safetensors.names() {
        if name.contains("moe.gate.weight") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if shape.len() == 2 {
                config.moe.num_experts = shape[0];
                config.moe.hidden_dim = shape[1];
                config.hidden_size = shape[1];
            }
            break;
        }
    }

    // Deduce layers count
    let mut max_layer_idx = 0;
    for name in safetensors.names() {
        if name.starts_with("layers.") {
            if let Some(rest) = name.strip_prefix("layers.") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx > max_layer_idx { max_layer_idx = idx; }
                    }
                }
            }
        }
    }
    config.num_layers = max_layer_idx + 1;

    // Deduce intermediate dim from per-expert gate_proj.weight packed shape
    for name in safetensors.names() {
        if name.ends_with("gate_proj.weight") && !name.contains("shared_experts") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if shape.len() == 2 {
                config.moe.intermediate_dim = shape[0];
                config.moe.hidden_dim = shape[1] * 4;
                config.hidden_size = shape[1] * 4;
            }
            break;
        }
    }

    // Deduce vocab size from embed/lm_head
    for name in safetensors.names() {
        if name.contains("embed") || name.contains("lm_head") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if !shape.is_empty() {
                config.vocab_size = shape[0];
                break;
            }
        }
    }

    // Deduce attention heads from q_proj
    for name in safetensors.names() {
        if name.ends_with("q_proj.weight") && name.contains("self_attn") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if shape.len() == 2 {
                let q_out = shape[0];
                // For DeepSeek MLA: q_proj output = num_heads * (qk_nope_dim + qk_rope_dim)
                // Common: num_heads = 16, qk_nope_dim = 128, qk_rope_dim = 64
                config.num_attention_heads = 16;
                config.qk_nope_head_dim = 128;
                config.qk_rope_head_dim = 64;
                // Verify: 16 * (128 + 64) = 3072
                debug_assert_eq!(q_out, config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim));
            }
            break;
        }
    }

    // Deduce kv_lora_rank from kv_a_proj_with_mqa
    for name in safetensors.names() {
        if name.contains("kv_a_proj_with_mqa") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if shape.len() == 2 {
                let kv_a_out = shape[0];
                config.kv_lora_rank = kv_a_out - config.qk_rope_head_dim;
                config.v_head_dim = config.qk_nope_head_dim;
            }
            break;
        }
    }

    // Deduce num_kv_heads from kv_b_proj
    for name in safetensors.names() {
        if name.contains("kv_b_proj.weight") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if shape.len() == 2 {
                let kv_b_out = shape[0];
                config.num_key_value_heads = kv_b_out / (config.qk_nope_head_dim + config.v_head_dim);
            }
            break;
        }
    }

    config.moe.top_k = 6;
    config.use_shared_experts = safetensors.names().iter().any(|n| n.contains("shared_experts"));

    Ok(config)
}

pub fn parse_config_json<P: AsRef<Path>>(path: P) -> Result<ModelConfig> {
    let file = File::open(path)?;
    let parsed: ConfigJson = serde_json::from_reader(file)?;

    let mut config = ModelConfig::default();
    config.vocab_size = parsed.vocab_size.unwrap_or(102400);
    config.hidden_size = parsed.hidden_size.unwrap_or(2048);
    config.num_layers = parsed.num_layers.or(parsed.num_hidden_layers).unwrap_or(27);
    config.moe.num_experts = parsed.num_experts.or(parsed.num_local_experts).unwrap_or(64);
    config.moe.top_k = parsed.top_k.or(parsed.num_experts_per_tok).unwrap_or(6);
    config.moe.intermediate_dim = parsed.intermediate_dim.or(parsed.intermediate_size).unwrap_or(1408);
    config.moe.hidden_dim = parsed.hidden_dim.unwrap_or(config.hidden_size);

    Ok(config)
}

/// Load per-expert weights for a single layer (format: layers.{l}.moe.experts.{e}.{proj}.*).
fn load_per_expert_experts(
    safetensors: &SafeTensors<'_>,
    layer_idx: usize,
    num_experts: usize,
    moe: &MoEConfig,
    device: &Device,
) -> Result<Vec<PackedExpert>> {
    use candle_core::safetensors::Load;
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gate_proj_key = format!("layers.{layer_idx}.moe.experts.{e}.gate_proj.weight");
        let up_proj_key = format!("layers.{layer_idx}.moe.experts.{e}.up_proj.weight");
        let down_proj_key = format!("layers.{layer_idx}.moe.experts.{e}.down_proj.weight");
        let gate_scales_key = format!("layers.{layer_idx}.moe.experts.{e}.gate_proj.scales");
        let up_scales_key = format!("layers.{layer_idx}.moe.experts.{e}.up_proj.scales");
        let down_scales_key = format!("layers.{layer_idx}.moe.experts.{e}.down_proj.scales");

        let gate_view = safetensors.tensor(&gate_proj_key)
            .with_context(|| format!("Missing expert gate weight: {gate_proj_key}"))?;
        let gate_scales_view = safetensors.tensor(&gate_scales_key)
            .with_context(|| format!("Missing expert gate scales: {gate_scales_key}"))?;
        let up_view = safetensors.tensor(&up_proj_key)
            .with_context(|| format!("Missing expert up weight: {up_proj_key}"))?;
        let up_scales_view = safetensors.tensor(&up_scales_key)
            .with_context(|| format!("Missing expert up scales: {up_scales_key}"))?;
        let down_view = safetensors.tensor(&down_proj_key)
            .with_context(|| format!("Missing expert down weight: {down_proj_key}"))?;
        let down_scales_view = safetensors.tensor(&down_scales_key)
            .with_context(|| format!("Missing expert down scales: {down_scales_key}"))?;

        let gate_scales = gate_scales_view.load(device)?;
        let up_scales = up_scales_view.load(device)?;
        let down_scales = down_scales_view.load(device)?;

        let gate_proj = PackedQMoETensor::from_bytes(
            gate_view.data().to_vec(),
            (moe.intermediate_dim, moe.hidden_dim),
            gate_scales,
        );
        let up_proj = PackedQMoETensor::from_bytes(
            up_view.data().to_vec(),
            (moe.intermediate_dim, moe.hidden_dim),
            up_scales,
        );
        let down_proj = PackedQMoETensor::from_bytes(
            down_view.data().to_vec(),
            (moe.hidden_dim, moe.intermediate_dim),
            down_scales,
        );

        experts.push(PackedExpert { gate_proj, up_proj, down_proj });
    }
    Ok(experts)
}

/// Load a flattened layer as a SharedExpert (always-on FFN, no per-expert routing).
fn load_flattened_as_shared(
    safetensors: &SafeTensors<'_>,
    layer_idx: usize,
    device: &Device,
) -> Result<SharedExpert> {
    use candle_core::safetensors::Load;

    let gate_proj_key = format!("layers.{layer_idx}.moe.gate_proj.weight");
    let up_proj_key = format!("layers.{layer_idx}.moe.up_proj.weight");
    let down_proj_key = format!("layers.{layer_idx}.moe.down_proj.weight");
    let gate_scales_key = format!("layers.{layer_idx}.moe.gate_proj.scales");
    let up_scales_key = format!("layers.{layer_idx}.moe.up_proj.scales");
    let down_scales_key = format!("layers.{layer_idx}.moe.down_proj.scales");

    let gate_view = safetensors.tensor(&gate_proj_key)
        .with_context(|| format!("Missing flattened gate weight: {gate_proj_key}"))?;
    let up_view = safetensors.tensor(&up_proj_key)
        .with_context(|| format!("Missing flattened up weight: {up_proj_key}"))?;
    let down_view = safetensors.tensor(&down_proj_key)
        .with_context(|| format!("Missing flattened down weight: {down_proj_key}"))?;

    let gate_scales = safetensors.tensor(&gate_scales_key)?.load(device)?;
    let up_scales = safetensors.tensor(&up_scales_key)?.load(device)?;
    let down_scales = safetensors.tensor(&down_scales_key)?.load(device)?;

    let gate_shape = gate_view.shape();
    let up_shape = up_view.shape();
    let down_shape = down_view.shape();

    Ok(SharedExpert {
        gate_proj: PackedQMoETensor::from_bytes(
            gate_view.data().to_vec(),
            (gate_shape[0], gate_shape[1] * 4),
            gate_scales,
        ),
        up_proj: PackedQMoETensor::from_bytes(
            up_view.data().to_vec(),
            (up_shape[0], up_shape[1] * 4),
            up_scales,
        ),
        down_proj: PackedQMoETensor::from_bytes(
            down_view.data().to_vec(),
            (down_shape[0], down_shape[1] * 4),
            down_scales,
        ),
    })
}

/// Load shared expert for a layer (format: layers.{l}.moe.shared_experts.{proj}.*).
fn load_shared_expert(
    safetensors: &SafeTensors<'_>,
    layer_idx: usize,
    device: &Device,
) -> Result<Option<SharedExpert>> {
    use candle_core::safetensors::Load;

    let gate_key = format!("layers.{layer_idx}.moe.shared_experts.gate_proj.weight");
    let up_key = format!("layers.{layer_idx}.moe.shared_experts.up_proj.weight");
    let down_key = format!("layers.{layer_idx}.moe.shared_experts.down_proj.weight");

    if !safetensors.names().iter().any(|n| *n == gate_key) {
        return Ok(None);
    }

    let gate_s = safetensors.tensor(&gate_key)?;
    let up_s = safetensors.tensor(&up_key)?;
    let down_s = safetensors.tensor(&down_key)?;

    let gate_scales_key = format!("layers.{layer_idx}.moe.shared_experts.gate_proj.scales");
    let up_scales_key = format!("layers.{layer_idx}.moe.shared_experts.up_proj.scales");
    let down_scales_key = format!("layers.{layer_idx}.moe.shared_experts.down_proj.scales");

    let gate_scales = safetensors.tensor(&gate_scales_key)?.load(device)?;
    let up_scales = safetensors.tensor(&up_scales_key)?.load(device)?;
    let down_scales = safetensors.tensor(&down_scales_key)?.load(device)?;

    let gate_shape = gate_s.shape();
    let up_shape = up_s.shape();
    let down_shape = down_s.shape();

    let shared = SharedExpert {
        gate_proj: PackedQMoETensor::from_bytes(
            gate_s.data().to_vec(),
            (gate_shape[0], gate_shape[1] * 4),
            gate_scales,
        ),
        up_proj: PackedQMoETensor::from_bytes(
            up_s.data().to_vec(),
            (up_shape[0], up_shape[1] * 4),
            up_scales,
        ),
        down_proj: PackedQMoETensor::from_bytes(
            down_s.data().to_vec(),
            (down_shape[0], down_shape[1] * 4),
            down_scales,
        ),
    };

    Ok(Some(shared))
}

/// Check if a layer uses per-expert expert format.
fn has_per_expert_format(safetensors: &SafeTensors<'_>, layer_idx: usize) -> bool {
    let key = format!("layers.{layer_idx}.moe.experts.0.gate_proj.weight");
    safetensors.names().iter().any(|n| *n == key)
}

/// Check if a layer has a gate weight.
fn has_gate_weight(safetensors: &SafeTensors<'_>, layer_idx: usize) -> bool {
    let key = format!("layers.{layer_idx}.moe.gate.weight");
    safetensors.names().iter().any(|n| *n == key)
}

pub fn load_model_from_safetensors<P: AsRef<Path>>(
    model_path: P,
    config_path: Option<std::path::PathBuf>,
    device: &Device,
) -> Result<(DeepSeekCoderV2, ModelConfig)> {
    let model_path = model_path.as_ref();
    tracing::info!("Opening safetensors model file at {:?}", model_path);

    let file = File::open(model_path)
        .with_context(|| format!("Failed to open model file {:?}", model_path))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?;

    let config = if let Some(ref path) = config_path {
        tracing::info!("Parsing config from explicit path {:?}", path);
        let mut cfg = parse_config_json(path)?;
        // Deduce MLA params from tensor shapes (they're not in the config file)
        let deduced = deduce_config(&safetensors)?;
        cfg.num_attention_heads = deduced.num_attention_heads;
        cfg.num_key_value_heads = deduced.num_key_value_heads;
        cfg.qk_nope_head_dim = deduced.qk_nope_head_dim;
        cfg.qk_rope_head_dim = deduced.qk_rope_head_dim;
        cfg.v_head_dim = deduced.v_head_dim;
        cfg.kv_lora_rank = deduced.kv_lora_rank;
        cfg.use_shared_experts = deduced.use_shared_experts;
        cfg
    } else {
        tracing::info!("No config file found. Deduce all hyperparameters from tensors...");
        deduce_config(&safetensors)?
    };

    tracing::info!("Configured Model: {:?}", config);

    // Load standard weights into a VarMap (skipping expert tensors)
    let mut varmap = VarMap::new();
    load_standard_weights(&safetensors, &mut varmap, device)?;

    // Handle gate weight: if layer 0 is flattened and has no gate.weight,
    // we need to ensure all layers have one. Create dummy if missing.
    {
        let mut varmap_data = varmap.data().lock().unwrap();
        for l in 0..config.num_layers {
            let key = format!("layers.{l}.moe.gate.weight");
            if !varmap_data.contains_key(&key) && has_gate_weight(&safetensors, l) {
                // Load gate weight separately (it's a standard weight that should have been loaded)
            } else if !varmap_data.contains_key(&key) {
                // Create a uniform gate weight for layers without one
                let uniform = Tensor::ones(
                    (config.moe.num_experts, config.hidden_size),
                    DType::F32,
                    device,
                )?;
                let var = Var::from_tensor(&uniform)?;
                varmap_data.insert(key, var);
            }
        }
    }

    // Load packed experts per layer
    tracing::info!("Loading packed expert weights...");
    let mut all_layers_experts = Vec::new();
    let mut all_layers_shared = Vec::new();
    for l in 0..config.num_layers {
        tracing::info!("  Loading layer {}/{} experts...", l + 1, config.num_layers);
        let (experts, shared) = if has_per_expert_format(&safetensors, l) {
            let experts = load_per_expert_experts(
                &safetensors, l, config.moe.num_experts, &config.moe, device,
            )?;
            let shared = load_shared_expert(&safetensors, l, device)?;
            (experts, shared)
        } else {
            // Flattened format (layer 0 typically) — treat as a shared expert, no routed experts
            let shared = Some(load_flattened_as_shared(&safetensors, l, device)?);
            (Vec::new(), shared)
        };
        all_layers_experts.push(experts);
        all_layers_shared.push(shared);
    }

    // Construct VarBuilder
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    // Build model
    let model = DeepSeekCoderV2::new(vb, &config, all_layers_experts, all_layers_shared)?;

    Ok((model, config))
}
