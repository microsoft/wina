import transformers
import torch
import torch.nn as nn
from transformers import AutoConfig
from collections import defaultdict

import os
import math
from utils.tensor_transform import tensor_transform

torch.set_printoptions(precision=10)

class SparsifyFn(nn.Module):
    def __init__(self, distr=None, init_sparsity=None, init_threshold=None, apply_prefill=True, sparse_mode=None, mask_by=None):
        super(SparsifyFn, self).__init__()

        assert init_sparsity is None or init_threshold is None, "init_sparsity and init_threshold cannot both be specified"

        if distr is not None and init_sparsity is not None:
            thresh = distr.icdf(0.5 + init_sparsity/2)
        elif init_threshold is not None:
            thresh = init_threshold
        else:
            init_sparsity = 0
            thresh = 0
        
        self.register_buffer("a", torch.tensor([thresh]).to(torch.float16))

        self.distr = distr
        self.apply_prefill = apply_prefill
        self.sparse_mode = sparse_mode
        self.mask_by = mask_by
        
        self.zero_count, self.numel = 0, 0

    def set_threshold(self, sparsity):
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        if self.mask_by == 'threshold':
            self.threshold = self.distr.icdf(sparsity).item() if sparsity != 0.0 else 0.0
        #self.threshold = self.distr.icdf(0.5 + sparsity/2).item() if sparsity != 0.0 else 0.0
        self.sparsity_level = sparsity
        self.actual_sparsity = sparsity

    def forward(self, x):

        # NOTE: we set the sparsify to 99% since the prefill sparsification phenomenon observed by TEAL
        assert self.mask_by in ['threshold', 'topk'], f"{self.mask_by} not supported"
        #x = x.to(torch.float16)
        if self.mask_by == 'threshold':
            #return self.apply(x)
            half_seq_len = max(1, int(0.99 * x.size(1)))
            last_context = x[:, -half_seq_len:, :]
            last_mask = self.apply(last_context)

            mask = torch.cat((torch.ones_like(x[:, :-half_seq_len, :]), last_mask), dim=1)
            
            # record actual_sparsity
            self.zero_count += torch.sum(last_mask == 0).item()
            self.numel += last_mask.numel()
            self.actual_sparsity = self.zero_count / self.numel
            
            return mask
        
        elif self.mask_by == 'topk':
            half_seq_len = max(1, int(0.99 * x.size(1)))
            last_x = x[:, -half_seq_len:, :]
            k = min(last_x.size(-1), max(1, int(last_x.size(-1) * (1-self.sparsity_level))))
            vals, topk_indices = last_x.abs().topk(k)
            last_mask = torch.zeros_like(last_x).scatter(-1, topk_indices, 1)
            mask = torch.cat((torch.ones_like(x[:, :-half_seq_len, :]), last_mask), dim=1)
            
            return mask

    def apply(self, x):
        return x.abs().gt(self.threshold) #* x
    
    def get_threshold(self):
        return self.threshold


def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)


class Distribution:
    def __init__(self, file_path, hidden_type):
        self.file_path = file_path
        self.hidden_type = hidden_type # h1 or h2 if sparse_mode == teal; k/q/v/o/gate/up/down if sparse_model == wina
        
        histogram = torch.load(f"{self.file_path}/histograms.pt", map_location="cpu", weights_only=True)

        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"].cpu(), histogram[self.hidden_type].cpu()

        self.total_count = self.counts.sum()
        
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    # kernel smoothing
    def pdf(self, x, bandwidth=None):
        if bandwidth is None:
            bandwidth =  1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        
        bin_centers = self.bin_centers.unsqueeze(1)
        
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        
        return pdf
    
    def cdf(self, x):
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    def kthvalue(self, k):
        return torch.kthvalue(self.activations, k).values
    
    # NOTE: Assumes distribution is zero mean unimodal
    def icdf(self, q):
        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        
        if idx == 0:
            return self.bin_centers[0]
        elif idx == len(self.bin_centers):
            return self.bin_centers[-1]
        else:
            lower_count = self.cumulative_counts[idx - 1]
            upper_count = self.cumulative_counts[idx]
            lower_value = self.bin_centers[idx - 1]
            upper_value = self.bin_centers[idx]
            
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)

class ActivationModule:
    def __init__(self, file_path):
        self.file_path = file_path
        self.activations = defaultdict(list)
        self.histograms = None
        
        self.store = {}
        
        self.histograms = {}
                        
    def grab_activations(self, x, key):
        if x.size(1) > 1:  # Check if seq_len > 1
            #self.activations[key].append(x.detach().squeeze(0).cpu().float())
            self.activations[key].append(x.to(torch.float16).abs().detach().squeeze(0).cpu())
    def save_activations(self):
        self.activations = self.combine_activations()
        torch.save(self.activations, f"{self.file_path}/activations.pt")

    def load_activations(self):
        self.activations = torch.load(f"{self.file_path}/activations.pt", map_location="cpu", weights_only=True)

    def find_threshold(self, q):
        threshold_dict = {}
        for key in self.projs:
            if q == 0:
                threshold_dict[key] = 0.0
                continue
            
            cumulative_counts = torch.cumsum(self.histograms[key], dim=0)
            
            target_count = q * self.histograms[key].sum()
            idx = torch.searchsorted(cumulative_counts, target_count)
            
            if idx == 0:
                threshold = self.histograms[f"{key}_centers"][0]
            elif idx == len(self.histograms[f"{key}_centers"]):
                threshold = self.histograms[f"{key}_centers"][-1]
            else:
                lower_count = cumulative_counts[idx - 1]
                upper_count = cumulative_counts[idx]
                lower_value = self.histograms[f"{key}_centers"][idx - 1]
                upper_value = self.histograms[f"{key}_centers"][idx]
                
                fraction = (target_count - lower_count) / (upper_count - lower_count)
                threshold = lower_value + fraction * (upper_value - lower_value)
            threshold_dict[key] = threshold.item()
        return threshold_dict        
    
    # NOTE: This doesn't store outlier activation values
    def find_histogram(self, num_bins=30000, head_outlier_threshold=0.0, tail_outlier_threshold=0.0):
        # for fine-grained analysis, do not combine activations
        self.activations = self.combine_activations()
        self.projs = self.activations.keys()
        torch.cuda.empty_cache()
        for key, acts in self.activations.items():
            torch.cuda.empty_cache()
            
            acts = acts.flatten().detach()
            
            if f"{key}_centers" not in self.histograms:
                bin_centers, counts = torch.unique(acts, sorted=True, return_inverse=False, return_counts=True, dim=None)
                self.histograms[f"{key}_centers"] = bin_centers.cpu()
                #self.histograms[key] = counts.to(torch.flaot32).cpu()
                self.histograms[key] = counts.float().cpu()
            else:                
                bin_centers, counts = torch.unique(acts, sorted=True, return_inverse=False, return_counts=True, dim=None)
                bin_centers, counts = bin_centers.cpu(), counts.cpu()
                merged_bin_centers = torch.unique(torch.cat([self.histograms[f"{key}_centers"], bin_centers]), sorted=False)
                mask = merged_bin_centers.unsqueeze(1) == self.histograms[f"{key}_centers"].unsqueeze(0)
                self.histograms[key] = (mask * self.histograms[key]).sum(dim=1)
                mask = merged_bin_centers.unsqueeze(1) == bin_centers.unsqueeze(0)
                self.histograms[key] += (mask * counts).sum(dim=1)

                self.histograms[f"{key}_centers"] = merged_bin_centers
                
        del acts
        self.activations = defaultdict(list)
        
        return self.histograms
    
    def save_histogram(self):
        os.makedirs(self.file_path, exist_ok=True)
        torch.save(self.histograms, f"{self.file_path}/histograms.pt")
        
    def combine_activations(self):
        combined_activations = {}
        for key, acts in self.activations.items():
            combined_activations[key] = torch.cat(acts, dim=0)
        return combined_activations

from transformers import AutoConfig

def get_model_class_name(model_name):
    try:
        # Fetch the model config
        config = AutoConfig.from_pretrained(model_name)
        
        # Get the model class name from the config
        model_class_name = config.architectures[0] if config.architectures else None
        
        return model_class_name
    except Exception as e:
        print(f"Error fetching model class name: {e}")
        return None


def get_sparse_model(model_name, device, histogram_path, sparse_mode, mask_by, torch_dtype=torch.float16, strategy_path=None, transform=None, **kwargs):
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

    class_name = get_model_class_name(model_name)

    assert class_name in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Phi3ForCausalLM", "LlamaSparseForCausalLM", "MistralSparseForCausalLM", "Qwen2SparseForCausalLM", "Phi3SparseForCausalLM"], f"Model class name {class_name} not supported"

    assert sparse_mode in ['wina', 'teal'], f"Sparse mode {sparse_mode} not supported"
    
    assert mask_by in ['topk', 'threshold'], f"Mask by {mask_by} not supported"
    
    if "Llama" in class_name:
        SparseModel = LlamaSparseForCausalLM
    elif "Mistral" in class_name:
        SparseModel = MistralSparseForCausalLM
    elif "Qwen2" in class_name:
        SparseModel = Qwen2SparseForCausalLM
    elif "Phi3" in class_name:
        SparseModel = Phi3SparseForCausalLM
    
    if device == 'auto':
        # multi gpu
        model = SparseModel.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto", histogram_path=histogram_path, sparse_mode=sparse_mode, mask_by=mask_by, strategy_path=strategy_path, **kwargs)
    else:
        model = SparseModel.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device, histogram_path=histogram_path, sparse_mode=sparse_mode, mask_by=mask_by, strategy_path=strategy_path, **kwargs)

    #transform tensor if needed
    if transform:
        tensor_transform(model)
        
    return model

def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def get_module_device(module):
    return next(module.parameters()).device




def get_layer_greedy_sparsities(layer_sparsities, results_dir):
    import pandas as pd
    num_layers = len(layer_sparsities)
    projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'] if 'phi' not in results_dir.lower() else ['qkv', 'o', 'gate_up', 'down']
    sparsities = {proj: [0.0] * num_layers for proj in projs}
    
    for layer, target_sparsity in enumerate(layer_sparsities):
        file_path = os.path.join(results_dir, f'layer-{layer}', 'results.csv')
        df = pd.read_csv(file_path)
        
        # Find the row with the closest effective sparsity
        closest_row = df.iloc[(df['Effective Sparsity'] - target_sparsity).abs().argsort()[:1]]
        
        for proj in projs:
            sparsities[proj][layer] = closest_row[proj].values[0]
    
    return sparsities


def get_actual_layer_sparsity(layer):
    if 'phi' in layer.__class__.__name__.lower():
        attn_proj = ['qkv_proj', 'o_proj']
        mlp_proj = ['gate_up_proj', 'down_proj']
        attn_proj_name = ['qkv', 'o']
        mlp_proj_name = ['gate_up', 'down']
    else:
        attn_proj = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        mlp_proj = ['gate_proj', 'up_proj', 'down_proj']
        attn_proj_name = ['q', 'k', 'v', 'o']
        mlp_proj_name = ['gate', 'up', 'down']
    attn_matrices_size = torch.tensor([dict(layer.self_attn.named_modules())[proj].weight.numel() for proj in attn_proj])
    mlp_matrices_size = torch.tensor([dict(layer.mlp.named_modules())[proj].weight.numel() for proj in mlp_proj])
    attn_matrices_sparsity = torch.tensor([layer.self_attn.sparse_fns[proj].actual_sparsity for proj in attn_proj_name])
    mlp_matrices_sparsity = torch.tensor([layer.mlp.sparse_fns[proj].actual_sparsity for proj in mlp_proj_name])
    sparsity = ((attn_matrices_sparsity * attn_matrices_size).sum() +  (mlp_matrices_sparsity * mlp_matrices_size).sum()) / (attn_matrices_size.sum() + mlp_matrices_size.sum())
    
    return sparsity.item()


def get_actual_model_sparsity(model):
    sparsity_list = []
    for layer in model.model.layers:
        sparsity_list.append(get_actual_layer_sparsity(layer))
    sparsity = sum(sparsity_list) / len(sparsity_list)
    return sparsity
