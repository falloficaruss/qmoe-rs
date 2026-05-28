use anyhow::{Context, Result};
use candle_core::{DType, Device, Var};
use candle_nn::{VarBuilder, VarMap};
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::path::Path;

use crate::model::{DeepSeekCoderV2, ModelConfig};
use crate::moe::{MoEConfig, PackedExpert};
use crate::tensor::PackedQMoETensor;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct ConfigJson {
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_local_experts: Option<usize>,
    pub top_k: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub hidden_dim: Option<usize>,
    pub intermediate_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
}

/// Helper function to load standard weights from safetensors to VarMap.
fn load_standard_weights(
    safetensors: &SafeTensors<'_>,
    varmap: &mut VarMap,
    device: &Device,
) -> Result<()> {
    use candle_core::safetensors::Load;

    let mut varmap_data = varmap.data().lock().unwrap();

    for name in safetensors.names() {
        // Skip expert weight matrices and scales since they are loaded manually as packed tensors
        if name.contains("moe.experts") && (name.contains(".weight") || name.contains(".scales")) {
            continue;
        }

        // Load standard weight tensor
        let view = safetensors.tensor(name)?;
        let tensor = view.load(device)?;
        let var = Var::from_tensor(&tensor)?;
        varmap_data.insert(name.to_string(), var);
    }

    Ok(())
}

/// Helper function to auto-deduce configuration from safetensors keys and shapes.
pub fn deduce_config(safetensors: &SafeTensors<'_>) -> Result<ModelConfig> {
    // Look for moe.gate.weight
    let mut num_experts = 4;
    let mut hidden_size = 64;

    // Check layer gating weights
    let gate_key = safetensors.names().iter()
        .find(|name| name.contains("moe.gate.weight"))
        .cloned();

    if let Some(key) = gate_key {
        let view = safetensors.tensor(&key)?;
        let shape = view.shape();
        if shape.len() == 2 {
            num_experts = shape[0];
            hidden_size = shape[1];
        }
    }

    // Deduce layers count by looking at the highest index in "layers.X."
    let mut max_layer_idx = 0;
    for name in safetensors.names() {
        if name.starts_with("layers.") {
            if let Some(rest) = name.strip_prefix("layers.") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx > max_layer_idx {
                            max_layer_idx = idx;
                        }
                    }
                }
            }
        }
    }
    let num_layers = max_layer_idx + 1;

    // Deduce intermediate_dim and hidden_dim
    // Look at layers.0.moe.experts.0.gate_proj.weight shape (or any expert weight)
    let mut intermediate_dim = 128;
    let mut hidden_dim = 64;

    let expert_proj_key = safetensors.names().iter()
        .find(|name| name.contains("gate_proj.weight"))
        .cloned();

    if let Some(key) = expert_proj_key {
        let view = safetensors.tensor(&key)?;
        let shape = view.shape();
        if shape.len() == 2 {
            intermediate_dim = shape[0];
            hidden_dim = shape[1] * 4; // Packed 4 weights per byte
        }
    }

    // Deduce vocab size if embed or lm_head weights are present
    let mut vocab_size = 102400;
    for name in safetensors.names() {
        if name.contains("embed") || name.contains("lm_head") {
            let view = safetensors.tensor(name)?;
            let shape = view.shape();
            if !shape.is_empty() {
                vocab_size = shape[0];
                break;
            }
        }
    }

    Ok(ModelConfig {
        vocab_size,
        hidden_size,
        num_layers,
        moe: MoEConfig {
            num_experts,
            top_k: 1, // Default top-k
            hidden_dim,
            intermediate_dim,
        },
    })
}

/// Parse configuration from a config.json file path
pub fn parse_config_json<P: AsRef<Path>>(path: P) -> Result<ModelConfig> {
    let file = File::open(path)?;
    let parsed: ConfigJson = serde_json::from_reader(file)?;

    let vocab_size = parsed.vocab_size.unwrap_or(102400);
    let hidden_size = parsed.hidden_size.unwrap_or(64);
    let num_layers = parsed.num_layers.or(parsed.num_hidden_layers).unwrap_or(1);

    let num_experts = parsed.num_experts.or(parsed.num_local_experts).unwrap_or(4);
    let top_k = parsed.top_k.or(parsed.num_experts_per_tok).unwrap_or(1);
    let intermediate_dim = parsed.intermediate_dim.or(parsed.intermediate_size).unwrap_or(128);
    let hidden_dim = parsed.hidden_dim.unwrap_or(hidden_size);

    Ok(ModelConfig {
        vocab_size,
        hidden_size,
        num_layers,
        moe: MoEConfig {
            num_experts,
            top_k,
            hidden_dim,
            intermediate_dim,
        },
    })
}

/// Load entire DeepSeekCoderV2 model from a .safetensors file path
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

    // 1. Resolve configuration (either config.json or auto-deduction)
    let config = if let Some(ref path) = config_path {
        tracing::info!("Parsing config from explicit path {:?}", path);
        parse_config_json(path)?
    } else {
        let sibling_config = model_path.parent()
            .map(|p| p.join("config.json"))
            .filter(|p| p.exists());

        if let Some(path) = sibling_config {
            tracing::info!("Found config.json next to model file at {:?}", path);
            parse_config_json(path)?
        } else {
            tracing::info!("No config file found. Deduce model hyperparameters from tensor shapes...");
            deduce_config(&safetensors)?
        }
    };

    tracing::info!("Configured Model: {:?}", config);

    // 2. Load standard high-precision weights into a VarMap
    let mut varmap = VarMap::new();
    load_standard_weights(&safetensors, &mut varmap, device)?;

    // If "moe.gate.weight" is present globally in the safetensors (like in our mock)
    // but not per layer, we duplicate it for all layers so the layers can retrieve it.
    let mut varmap_data = varmap.data().lock().unwrap();
    if varmap_data.contains_key("moe.gate.weight") {
        let global_gate = varmap_data.get("moe.gate.weight").unwrap().clone();
        for l in 0..config.num_layers {
            let key = format!("layers.{}.moe.gate.weight", l);
            if !varmap_data.contains_key(&key) {
                varmap_data.insert(key, global_gate.clone());
            }
        }
    }
    drop(varmap_data);

    // 3. Load packed experts manually
    tracing::info!("Loading packed expert weights...");
    let mut all_layers_experts = Vec::new();
    for l in 0..config.num_layers {
        let mut layer_experts = Vec::new();
        for e in 0..config.moe.num_experts {
            // Retrieve byte projections
            let gate_proj_key = format!("layers.{}.moe.experts.{}.gate_proj.weight", l, e);
            let up_proj_key = format!("layers.{}.moe.experts.{}.up_proj.weight", l, e);
            let down_proj_key = format!("layers.{}.moe.experts.{}.down_proj.weight", l, e);

            // Retrieve scales
            let gate_scales_key = format!("layers.{}.moe.experts.{}.gate_proj.scales", l, e);
            let up_scales_key = format!("layers.{}.moe.experts.{}.up_proj.scales", l, e);
            let down_scales_key = format!("layers.{}.moe.experts.{}.down_proj.scales", l, e);

            // Load gate_proj
            let gate_view = safetensors.tensor(&gate_proj_key)
                .with_context(|| format!("Missing packed expert gate weight: {}", gate_proj_key))?;
            let gate_scales_view = safetensors.tensor(&gate_scales_key)
                .with_context(|| format!("Missing expert gate scale: {}", gate_scales_key))?;
            
            use candle_core::safetensors::Load;
            let gate_scales = gate_scales_view.load(device)?;
            let gate_proj = PackedQMoETensor::from_bytes(
                gate_view.data().to_vec(),
                (config.moe.intermediate_dim, config.moe.hidden_dim),
                gate_scales,
            );

            // Load up_proj
            let up_view = safetensors.tensor(&up_proj_key)
                .with_context(|| format!("Missing packed expert up weight: {}", up_proj_key))?;
            let up_scales_view = safetensors.tensor(&up_scales_key)
                .with_context(|| format!("Missing expert up scale: {}", up_scales_key))?;
            let up_scales = up_scales_view.load(device)?;
            let up_proj = PackedQMoETensor::from_bytes(
                up_view.data().to_vec(),
                (config.moe.intermediate_dim, config.moe.hidden_dim),
                up_scales,
            );

            // Load down_proj
            let down_view = safetensors.tensor(&down_proj_key)
                .with_context(|| format!("Missing packed expert down weight: {}", down_proj_key))?;
            let down_scales_view = safetensors.tensor(&down_scales_key)
                .with_context(|| format!("Missing expert down scale: {}", down_scales_key))?;
            let down_scales = down_scales_view.load(device)?;
            let down_proj = PackedQMoETensor::from_bytes(
                down_view.data().to_vec(),
                (config.moe.hidden_dim, config.moe.intermediate_dim),
                down_scales,
            );

            layer_experts.push(PackedExpert {
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        all_layers_experts.push(layer_experts);
    }

    // 4. Construct VarBuilder from VarMap for standard weights
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    // 5. Build DeepSeekCoderV2 structure
    let model = DeepSeekCoderV2::new(vb, &config, all_layers_experts)?;

    Ok((model, config))
}
