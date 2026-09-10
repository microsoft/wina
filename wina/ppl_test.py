import sys,os
# sys.path.append('../')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import torch
from tqdm import tqdm
import os
import argparse

import json

from utils.utils import get_actual_model_sparsity

import pandas as pd

from utils.tensor_transform import tensor_transform
from transformers import LlamaForCausalLM

if __name__ == "__main__":
    from utils.utils import get_tokenizer, get_sparse_model
    from utils.eval_ppl import eval_ppl

    from wina.model import LlamaSparseForCausalLM, LlamaSparseConfig
    from wina.model import MistralSparseForCausalLM, MistralSparseConfig

    from utils.data import get_dataset

    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("llama_sparse", LlamaSparseConfig)
    AutoConfig.register("mistral_sparse", MistralSparseConfig)

    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
    AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output')
    parser.add_argument('--greedy', default=False, action='store_true', help='Flag for greedy')
    parser.add_argument('--sparsity', type=float, required=True, help='Sparsity level')
    parser.add_argument('--sparse_mode', type=str, required=True, help="Sparse approach, can be 'wina' or 'teal'")
    parser.add_argument('--mask_by', type=str, default='topk', help="Element selection method, can be 'topk' or 'threshold'")
    parser.add_argument('--transform', action='store_true', default=False, help='Flag for tensor transformation')
    parser.add_argument('--sample_size', type=int, default=1000, help='Size of samples to compute PPL')
    parser.add_argument('--seed', type=int, default=42, help='Seed for dataset shuffle')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*40)
    print("Start evaluating PPL")
    print('sparsity:', args.sparsity)

    tokenizer = get_tokenizer(args.model_name)
    model = get_sparse_model(args.model_name, device=device,
                            histogram_path=os.path.join(args.output_path, "histograms"), 
                            sparse_mode=args.sparse_mode, mask_by=args.mask_by,
                            transform=args.transform
                            )
    
    if device == "cuda":
        model.bfloat16()

    print("sample size:",args.sample_size)
    dataset = get_dataset(
        "wikitext",
        subset=None,
        split='test',
        size=args.sample_size, 
        seed=args.seed,
    )
    
    if args.greedy:
        print("Evaluating greedy PPL")
        greedy_path = os.path.join(args.output_path, "lookup")
        model.load_greedy_sparsities(greedy_path, args.sparsity)
    else:
        print("Evaluating uniform PPL")
        model.set_uniform_sparsity(args.sparsity)

    sparse_ppl = eval_ppl(model, tokenizer, device=device, dataset=dataset, debug=False)
    print(f"PPL: {sparse_ppl}")
    
    # record actual sparsity
    actual_sparsity = get_actual_model_sparsity(model)
    
    # save results
    result_dir = os.path.join(args.output_path, "results", args.mask_by, str(args.sparsity))
    os.makedirs(result_dir, exist_ok=True)
    file = os.path.join(result_dir, "ppl_results.json")
    content = {
        'model':{
            'model_name': args.model_name,
            'sparse_mode': args.sparse_mode,
            'greedy': args.greedy,
            'mask_by': args.mask_by,
            'transform': args.transform,
            'target_sparsity': args.sparsity,
            'actual_sparsity': actual_sparsity,
        },
        'dataset':{
            'sample_size': args.sample_size,
            'seed': args.seed
        },
        'ppl': sparse_ppl
        }
    
    with open(file, 'w') as f:
        json.dump(content, f, indent=4)
    
    print('Finish evaluating PPL')
    print(f'The result has been saved in {result_dir}')
    print("="*40)
