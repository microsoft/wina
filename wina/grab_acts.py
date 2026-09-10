# should get data, tokenizer, then layer by layer grab activations and save histograms + the activations themselves



import argparse
import os

import torch
import transformers

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))


from utils.utils import get_tokenizer, get_sparse_model

from wina.model import LlamaSparseForCausalLM, LlamaSparseConfig
from wina.model import MistralSparseForCausalLM, MistralSparseConfig

from transformers import AutoConfig, AutoModelForCausalLM
from utils.tensor_transform import tensor_transform

AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoConfig.register("mistral_sparse", MistralSparseConfig)

AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
parser.add_argument('--model_name', type=str, required=True,help='Path of the model to use')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output')
parser.add_argument('--sparse_mode', type=str, required=True, help="Sparsity method, can be 'wina' or 'teal")
parser.add_argument('--sparsity', type=float, help="Sparsity of the model")
parser.add_argument('--transform', action='store_true', default=False, help="Flag for tensor transformation")
parser.add_argument('--layer_wise', action='store_true', default=False, help="Flag for constructing histogram layer by layer")
parser.add_argument('--sample_size', type=int, default=20, help="Size of samples for constructing histogram")
parser.add_argument('--dataset_name', type=str, default='alpaca', help="Dataset for constructing histogram, can be alpaca/wikitext/commonsense")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*40)
print('Start threshold calculating')
print('model name:', args.model_name)
print('sparsity:', args.sparsity)

tokenizer = get_tokenizer(args.model_name)
model = get_sparse_model(args.model_name, device="cpu", 
                        histogram_path=os.path.join(args.output_path, "histograms"), 
                        grab_acts=True, sparse_mode=args.sparse_mode, 
                        mask_by='threshold', 
                        transform=args.transform)

from utils.data import get_dataset
from tqdm import tqdm
import gc
import csv

dataset = get_dataset(
    args.dataset_name,
    subset=None,
    split="train",
    size=args.sample_size,
    start=0
)

seq_len = 2048

def tokenize(sample):
    return tokenizer(sample['text'], truncation=True, max_length=seq_len, padding=False)
encodings = dataset.map(tokenize).remove_columns(["instruction", "input", "output", "text"])

hidden_states_all = []
position_ids_all = []
for input_ids in encodings['input_ids']:
    hidden_states = model.model.embed_tokens(torch.tensor(input_ids).unsqueeze(0))
    position_ids = torch.arange(hidden_states.size(1), dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    hidden_states_all.append(hidden_states)
    position_ids_all.append(position_ids)

attention_mask = None
past_key_value=None
output_attentions = False
use_cache = False
cache_position=None


act_path = os.path.join(args.output_path, "activations")
os.makedirs(act_path, exist_ok=True)


    
threshold_path = os.path.join(args.output_path, "thresholds")
#os.makedirs(threshold_path, exist_ok=True)

for layer_idx in tqdm(range(len(model.model.layers))):
    layer = model.model.layers[layer_idx].to(device)
        
    # save hidden_states for greedyopt
    if not args.layer_wise:
        torch.save(hidden_states_all[:20], os.path.join(act_path, f"act_{layer_idx}.pt"))
    
    # grab hidden states and construct histogram
    for sample_idx in range(len(hidden_states_all)):
        hidden_states, position_ids = hidden_states_all[sample_idx], position_ids_all[sample_idx]
        output = layer(hidden_states.to(device), attention_mask, position_ids.to(device), past_key_value, output_attentions, use_cache, cache_position)[0].cpu()
        if not args.layer_wise:
            # if not layer-wise: update hidden_states 
            hidden_states_all[sample_idx] = output
        
        # construct histogram and save thresholds
        if sample_idx > 0 and (sample_idx % 100 == 0 or sample_idx == len(hidden_states_all)-1):
            layer.mlp.activation_module.find_histogram()
            layer.self_attn.activation_module.find_histogram()
            torch.cuda.empty_cache()
    
    # save final histogram
    layer.mlp.activation_module.save_histogram()
    layer.self_attn.activation_module.save_histogram()
    
    # if layer_wise: set thresholds and update hidden states
    if args.layer_wise:
        layer.mlp.add_sparse_fns(sparsity = args.sparsity, mask_by='threshold')
        layer.self_attn.add_sparse_fns(sparsity = args.sparsity, mask_by='threshold')
        attn_proj = ['qkv','o'] if 'phi' in layer.__class__.__name__.lower() else ['q','k','v','o']
        mlp_proj = ['gate_up','down'] if 'phi' in layer.__class__.__name__.lower() else ['gate','up','down']
        for proj in attn_proj:
            layer.self_attn.sparse_fns[proj].set_threshold(args.sparsity)
        for proj in mlp_proj:
            layer.mlp.sparse_fns[proj].set_threshold(args.sparsity)
        for sample_idx in range(len(hidden_states_all)):
            hidden_states, position_ids = hidden_states_all[sample_idx], position_ids_all[sample_idx]
            hidden_states = layer(hidden_states.to(device), attention_mask, position_ids.to(device), past_key_value, output_attentions, use_cache, cache_position)[0].cpu()
            hidden_states_all[sample_idx] = hidden_states    

    del layer.mlp.activation_module.activations
    del layer.self_attn.activation_module.activations
    
    model.model.layers[layer_idx] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
print("Finish threshold calculating")
print('='*40)
