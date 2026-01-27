import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig


def quantize_tensor(weight, bits=8):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    scale = weight.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    
    quantized = (weight / scale).round().clamp(qmin, qmax)
    
    return quantized.to(torch.int8), scale.item()


def dequantize_tensor(quantized, scale):
    return quantized.float() * scale


def quantize_linear(linear_layer, bits=8):
    weight = linear_layer.weight.data
    quantized, scale = quantize_tensor(weight, bits)
    return quantized, scale


def quantize_model(model, bits=8):
    quantized_state = {}
    scales = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            quantized, scale = quantize_tensor(param.data, bits)
            quantized_state[name] = quantized
            scales[name] = scale
        else:
            quantized_state[name] = param.data
    
    for name, buffer in model.named_buffers():
        quantized_state[name] = buffer.data
    
    return quantized_state, scales


def apply_quantization(model, scales, bits=8):
    for name, param in model.named_parameters():
        if name in scales and len(param.shape) >= 2:
            scale = scales[name]
            quantized = param.data
            param.data = dequantize_tensor(quantized, scale)


def export_int8(model, output_path, bits=8):
    quantized_state, scales = quantize_model(model, bits)
    
    export_data = {
        'quantized_weights': quantized_state,
        'scales': scales,
        'config': {
            'n_layer': model.config.n_layer,
            'n_head': model.config.n_head,
            'n_kv_head': model.config.n_kv_head,
            'n_embd': model.config.n_embd,
            'vocab_size': model.config.vocab_size,
            'sequence_len': model.config.sequence_len,
            'window_pattern': model.config.window_pattern,
            'use_mqa': model.config.use_mqa,
            'multi_token_n': model.config.multi_token_n,
            'draft_n': model.config.draft_n,
            'draft_hidden_mult': model.config.draft_hidden_mult,
        },
        'bits': bits,
    }
    
    torch.save(export_data, output_path)
    return export_data


def load_int8(model_path, device):
    data = torch.load(model_path, map_location=device)
    
    config_kwargs = data['config']
    config = GPTConfig(**config_kwargs)
    
    with torch.device("meta"):
        model = GPT(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    quantized_state = data['quantized_weights']
    scales = data['scales']
    
    state_dict = {}
    for name, param in model.named_parameters():
        if name in quantized_state:
            if name in scales:
                quantized = quantized_state[name]
                scale = scales[name]
                state_dict[name] = dequantize_tensor(quantized, scale)
            else:
                state_dict[name] = quantized_state[name]
    
    for name, buffer in model.named_buffers():
        if name in quantized_state:
            state_dict[name] = quantized_state[name]
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, data['config'], scales

