import torch
import numpy as np
from safetensors.torch import save_file
import argparse
import os

def quantize_and_pack(weight, scale):
    """
    Quantizes floating-point weights to ternary values (-1, 0, 1) and packs 
    four weights into a single uint8 byte to match the Rust SIMD layout.
    
    Ternary Mapping:
      -1.0 -> 0 (binary 00)
       0.0 -> 1 (binary 01)
       1.0 -> 2 (binary 10)
    """
    # 1. Scale weights and round to nearest integer within [-1, 1]
    scaled = weight / scale[:, None]
    quantized = torch.clamp(torch.round(scaled), -1, 1).to(torch.int32)
    
    # Shift values from [-1, 0, 1] to [0, 1, 2] for unsigned byte representation
    mapped = quantized + 1
    
    out_features, in_features = mapped.shape
    assert in_features % 4 == 0, f"Input features ({in_features}) must be divisible by 4 for SIMD packing."
    
    # Reshape to group every 4 consecutive weights: [Out, In / 4, 4]
    grouped = mapped.view(out_features, in_features // 4, 4)
    
    # Pack 4 weights: [w3(2b) | w2(2b) | w1(2b) | w0(2b)]
    packed = (grouped[..., 3] << 6) | (grouped[..., 2] << 4) | (grouped[..., 1] << 2) | grouped[..., 0]
    
    return packed.to(torch.uint8)

def main():
    parser = argparse.ArgumentParser(description="Offline Python Quantization Pipeline for QMoE Rust Inference Engine")
    parser.add_argument("--input", type=str, help="Path to input standard PyTorch weight file (.bin or .pt)", default=None)
    parser.add_argument("--output", type=str, help="Path to save packed .safetensors file", default="qmoe_packed_model.safetensors")
    args = parser.parse_args()

    print("--- Starting QMoE Offline Quantization & Packing ---")
    
    # Generate dummy standard FP16 weights representing a DeepSeek-Coder-V2 layer if no input is provided
    if args.input is None or not os.path.exists(args.input):
        print("No valid input file provided. Generating mock DeepSeek-Coder-V2 weights for demonstration...")
        
        # DeepSeek-Coder-V2 layer dimension setup (simplified)
        num_layers = 2
        num_experts = 4
        hidden_dim = 64
        intermediate_dim = 128
        vocab_size = 102400
        
        state_dict = {}
        
        # Embedding
        state_dict["embed.weight"] = torch.randn(vocab_size, hidden_dim)
        
        # Final layer norm
        state_dict["norm.weight"] = torch.randn(hidden_dim)
        state_dict["norm.bias"] = torch.randn(hidden_dim)
        
        # LM head
        state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_dim)
        
        for l in range(num_layers):
            # Gating network
            state_dict[f"layers.{l}.moe.gate.weight"] = torch.randn(num_experts, hidden_dim)
            
            # Attention projections
            state_dict[f"layers.{l}.self_attn.q_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            state_dict[f"layers.{l}.self_attn.k_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            state_dict[f"layers.{l}.self_attn.v_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            state_dict[f"layers.{l}.self_attn.o_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
            
            # Layer norms
            state_dict[f"layers.{l}.input_layernorm.weight"] = torch.randn(hidden_dim)
            state_dict[f"layers.{l}.input_layernorm.bias"] = torch.randn(hidden_dim)
            state_dict[f"layers.{l}.post_attention_layernorm.weight"] = torch.randn(hidden_dim)
            state_dict[f"layers.{l}.post_attention_layernorm.bias"] = torch.randn(hidden_dim)
            
            # Populate mock expert projections
            for e in range(num_experts):
                state_dict[f"layers.{l}.moe.experts.{e}.gate_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
                state_dict[f"layers.{l}.moe.experts.{e}.up_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
                state_dict[f"layers.{l}.moe.experts.{e}.down_proj.weight"] = torch.randn(hidden_dim, intermediate_dim)
    else:
        print(f"Loading weights from {args.input}...")
        state_dict = torch.load(args.input)

    packed_dict = {}
    
    for key, tensor in state_dict.items():
        if "gate_proj.weight" in key or "up_proj.weight" in key or "down_proj.weight" in key:
            print(f"Quantizing and packing layer: {key}...")
            
            # Simple standard deviation scaling factor per row
            scale = tensor.std(dim=1)
            # Prevent division by zero
            scale = torch.clamp(scale, min=1e-5)
            
            packed_weights = quantize_and_pack(tensor, scale)
            
            packed_dict[key] = packed_weights
            packed_dict[key.replace(".weight", ".scales")] = scale
        else:
            # Gating weights remain in high-precision (e.g. Float32 / Float16)
            print(f"Keeping high-precision layer unchanged: {key}...")
            packed_dict[key] = tensor.to(torch.float32)

    print(f"Saving packed weights to {args.output}...")
    save_file(packed_dict, args.output)
    print("--- Successfully completed offline quantization pipeline! ---")

if __name__ == "__main__":
    main()
