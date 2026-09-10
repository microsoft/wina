
import os
import sys
import argparse
import torch
import csv
from copy import deepcopy


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from wina.model import (
    LlamaSparseForCausalLM, 
    LlamaSparseConfig,
    MistralSparseForCausalLM, 
    MistralSparseConfig, 
    Qwen2SparseConfig,
    Qwen2SparseForCausalLM,
    Phi3SparseConfig,
    Phi3SparseForCausalLM,
)
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)
AutoConfig.register("qwen2_sparse", Qwen2SparseConfig)
AutoModelForCausalLM.register(Qwen2SparseConfig, Qwen2SparseForCausalLM)
AutoConfig.register("phi3_sparse", Phi3SparseConfig)
AutoModelForCausalLM.register(Phi3SparseConfig, Phi3SparseForCausalLM)      

from utils.utils import get_sparse_model, get_tokenizer

import torch.cuda

torch.set_printoptions(precision=10)

weight_dict = {
    "Llama-3-8B": {
        'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Llama-2-7B": {
        'q': 1, 'k': 1, 'v': 1, 'o': 1,
        'gate': 2.6875, 'up': 2.6875, 'down': 2.6875
    },
    "Llama-2-13B": {
        'q': 1, 'k': 1, 'v': 1, 'o': 1,
        'gate': 2.7, 'up': 2.7, 'down': 2.7
    },
    "Mistral-7B": {
        'q': 1, 'k': 1/4, 'v': 1/4, 'o': 1,
        'gate': 3.5, 'up': 3.5, 'down': 3.5
    },
    "Qwen-2.5-7B": {
        'q': 1, 'k': 1/7, 'v': 1/7, 'o': 1,
        'gate': 5.2857, 'up': 5.2857, 'down': 5.2857
    },
    "Phi-4-14B": {
        'qkv': 1.5, 'o': 1,
        'gate_up': 7, 'down': 3.5
    }
}

def set_layer_sparsities(layer, sparsities):
    if 'phi' in layer.__class__.__name__.lower():
        layer.mlp.sparse_fns['gate_up'].set_threshold(sparsities['gate_up'])
        layer.mlp.sparse_fns['down'].set_threshold(sparsities['down'])
        
        layer.self_attn.sparse_fns['qkv'].set_threshold(sparsities['qkv'])
        layer.self_attn.sparse_fns['o'].set_threshold(sparsities['o'])
    else:
        layer.mlp.sparse_fns['gate'].set_threshold(sparsities['gate'])
        layer.mlp.sparse_fns['up'].set_threshold(sparsities['up'])
        layer.mlp.sparse_fns['down'].set_threshold(sparsities['down'])

        layer.self_attn.sparse_fns['q'].set_threshold(sparsities['q'])
        layer.self_attn.sparse_fns['k'].set_threshold(sparsities['k'])
        layer.self_attn.sparse_fns['v'].set_threshold(sparsities['v'])
        layer.self_attn.sparse_fns['o'].set_threshold(sparsities['o'])

def f(sparsities, weights):
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("weights must have a positive total")
    weighted_sparsity_sum = 0
    for projection_type, value in sparsities.items():
        if projection_type in weights:
            weighted_sparsity_sum += value * weights[projection_type]

    # print(weighted_sparsity_sum / total_weight)
    return weighted_sparsity_sum / total_weight

def layer_forward(layer, hidden_states):
    bsz, seq_len, _ = hidden_states.shape
    # layer = model.model.layers[layer_idx]

    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None

    return layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]


def calculate_activation_error(target_acts, new_activations, last_fraction=0.25):
    start_idx = int(new_activations.shape[1] * (1 - last_fraction))
    res = torch.norm(target_acts[:, start_idx:] - new_activations[:, start_idx:], dim=1).mean()
    #res = torch.norm(target_acts - new_activations, dim=-1).mean()
    return res

def calculate_baseline_error(layer, input_acts, target_acts, baseline_sparsities, last_fraction):
    set_layer_sparsities(layer, baseline_sparsities)
    new_activations = layer_forward(layer, input_acts)
    return calculate_activation_error(target_acts, new_activations, last_fraction)

def process_layer(layer, model_type, layer_idx, target_sparsity, base_step_size, last_fraction, output_path, sample_size):
    if model_type not in weight_dict:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    if not 0 <= target_sparsity <= 1:
        raise ValueError("target_sparsity must be in [0, 1]")
    if base_step_size <= 0:
        raise ValueError("base_step_size must be positive")
    if not 0 < last_fraction <= 1:
        raise ValueError("last_fraction must be in (0, 1]")
    weights = weight_dict[model_type]

    histogram_path = os.path.join(output_path, 'histograms')
    activations_path = os.path.join(output_path, 'activations', f'act_{layer_idx}.pt')
    output_path = os.path.join(output_path, 'lookup', f'layer-{layer_idx}', 'results.csv')

    device = next(layer.parameters()).device
    input_acts = torch.load(activations_path, map_location='cpu', weights_only=True)
    input_acts = input_acts[:sample_size]
    layer = layer.to(device)

    if 'phi' in layer.__class__.__name__.lower():
        projs = ['qkv', 'o', 'gate_up', 'down']
    else:
        projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    sparsities = {proj: 0.0 for proj in projs}
    set_layer_sparsities(layer, sparsities)
    #target_acts = layer_forward(layer, input_acts)
    target_acts = []
    for sample in input_acts:
        target_acts.append(layer_forward(layer, sample.to(device)))


    step_sizes = {proj: base_step_size * (1 / weights[proj]) for proj in projs}
    
    sparsities = {proj: 0.0 for proj in projs}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Effective Sparsity'] + ['Activation Error', 'Baseline Error'] + [f"{proj}" for proj in projs])

        while f(sparsities, weights) < target_sparsity:
            best_error = float('inf')
            best_proj = None
            
            for proj in projs:
                temp_sparsities = deepcopy(sparsities)

                if temp_sparsities[proj] >= 1:
                    continue

                temp_sparsities[proj] += step_sizes[proj]
                
                set_layer_sparsities(layer, temp_sparsities)
                
                error = 0
                for sample, target in zip(input_acts, target_acts):
                    new_activations = layer_forward(layer, sample.to(device))
                    error += calculate_activation_error(target, new_activations, last_fraction)
                
                if error < best_error:
                    best_error = error
                    best_proj = proj
            
            if best_proj is None:
                raise RuntimeError("Unable to increase sparsity; all projections reached their limits")
            sparsities[best_proj] = min(1.0, sparsities[best_proj] + step_sizes[best_proj])
            set_layer_sparsities(layer, sparsities)
            
            effective_sparsity = f(sparsities, weights)
            baseline_sparsities = {proj: effective_sparsity for proj in projs}
            baseline_error = 0
            for sample, target in zip(input_acts, target_acts):
                baseline_error += calculate_baseline_error(layer, sample.to(device), target, baseline_sparsities, last_fraction)
            
            row = [effective_sparsity] + [best_error.item(), baseline_error.item()] + [sparsities[proj] for proj in projs]
            csvwriter.writerow(row)
            csvfile.flush()
            
            print(f"Updated: Effective Sparsity: {effective_sparsity:.4f}, Activation Error: {best_error:.4f}, Baseline Error: {baseline_error:.4f}")
    
    layer.to('cpu')
    return sparsities

        
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path to the base model")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory") 
    parser.add_argument("--sparse_mode", type=str, required=True, help="Sparsity method, can be 'wina' or 'teal")
    parser.add_argument("--target_sparsity", type=float, default=0.66, help="Target effective sparsity")
    parser.add_argument("--transform", action='store_true', default=False, help="Flag for tensor transformation")
    parser.add_argument("--base_step_size", type=float, default=0.025, help="Base step size for sparsity updates")
    parser.add_argument("--last_fraction", type=float, default=0.25, help="Fraction of sequence to use for error calculation")
    parser.add_argument("--sample_size", type=int, default=20, help='Size of samples to compute the output error')
    
    args = parser.parse_args()
    
    print("="*40)
    print("Start calculating sparsity for each layer")

    histogram_path = os.path.join(args.output_path, 'histograms')


    from utils.utils import get_model_class_name

    class_name = get_model_class_name(args.model_name)
    assert class_name in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Phi3ForCausalLM", "LlamaSparseForCausalLM", "MistralSparseForCausalLM", "Qwen2SparseForCausalLM", "Phi3SparseForCausalLM"], f"Model class name {class_name} not supported"

    if "Llama" in class_name:
        SparseModel = LlamaSparseForCausalLM
    elif "Mistral" in class_name:
        SparseModel = MistralSparseForCausalLM
    elif "Qwen2" in class_name:
        SparseModel = Qwen2SparseForCausalLM
    elif "Phi3" in class_name:
        SparseModel = Phi3SparseForCausalLM
    model = get_sparse_model(args.model_name, device='cpu', histogram_path=histogram_path, sparse_mode=args.sparse_mode, mask_by='threshold', transform=args.transform)

    os.makedirs(os.path.join(args.output_path, 'lookup'), exist_ok=True)

    num_layers = len(model.model.layers)
     
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}")
        layer = model.model.layers[layer_idx]
        process_layer(layer, args.model_type, layer_idx, args.target_sparsity, args.base_step_size, args.last_fraction, args.output_path, args.sample_size)

    print("Finish calculating sparsity for each layer")
    print("="*40)
