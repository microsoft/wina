import argparse
import os
import torch
from utils.utils import get_sparse_model
import json


def eval(args):
    args.curr_mode = 'lm_eval'
    import lm_eval_harness
    info = lm_eval_harness.proceed(args)
    if info is None:
        raise RuntimeError("The evaluation harness returned no results")
    result = {'actual_sparsity': info.get('actual_sparsity')}
    for task in args.eval_tasks:
            if task not in info.get('results', {}):
                continue
            if 'acc_norm,none' in info['results'][task]:
                result[task] = "%.4f" % (info['results'][task]['acc_norm,none'])
            elif 'exact_match,flexible-extract' in info['results'][task]:
                result[task] = "%.4f" % (info['results'][task]['exact_match,flexible-extract'])  
            elif 'acc,none' in info['results'][task]:
                result[task] = "%.4f" % (info['results'][task]['acc,none'])  
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--base_model', type=str, required=True, help='Path to load the model')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results')
    parser.add_argument('--sparsity', type=float, required=True, help="Sparsity of the model")
    parser.add_argument('--sparse_mode', type=str, required=True, help='Sparse approach, can be wina or teal')
    parser.add_argument('--eval_tasks', type=lambda value: [item.strip() for item in value.split(',') if item.strip()], required=False, default=['piqa','arc_challenge','winogrande','hellaswag','mmlu','gsm8k'])
    parser.add_argument('--greedy', action="store_true", default=False, help='Flags for greedy')
    parser.add_argument('--mask_by', type=str, default='topk', help="Element selection strategy, can be 'topk' or 'threshold'")
    parser.add_argument('--transform', action='store_true', default=False, help='Flag for tensor transformation')
    #parser.add_argument('--eval_datasets', type=list, default=['wikitext2'])
    parser.add_argument('--eval_device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    print('='*40)
    print("Start runing lm-eval harness")
    print("model name:", args.base_model)
    print("sparse mode", args.sparse_mode)
    
    result = eval(args)
    
    # save results
    result_dir = os.path.join(args.save_path, "results", args.mask_by, str(args.sparsity))
    os.makedirs(result_dir, exist_ok=True)
    file = os.path.join(result_dir, 'lm_eval_results.json')
    content = {
        'model': {
            'model_name': args.base_model,
            'sparse_mode': args.sparse_mode,
            'sparsity': args.sparsity,
            'greedy': args.greedy,
            'mask_by': args.mask_by,
            'transform': args.transform
        },
        'result': result}
    with open(file, 'w') as f:
        json.dump(content, f, indent=4)

    print("Finish Eval of Sparsity: ", args.sparsity)
    print(f"The result has been saved in {result_dir}")
    print('='*40)
