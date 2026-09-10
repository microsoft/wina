import torch
from torch import nn
from torch.nn import Linear, Module, Parameter
from typing import Iterable
import numpy as np
import gc
import inspect
import logging
from typing import TypeVar
from tqdm import tqdm
from wina.decoder_layer import _monkeypatch_decoder


def get_first_layernorm(layer) -> Module:
    return layer.input_layernorm

def get_second_layernorm(layer) -> Module:
    return layer.post_attention_layernorm

def get_attention_inputs(layer) -> list[Linear]:
    if 'phi' in layer.__class__.__name__.lower():
        return [layer.self_attn.qkv_proj]
    else:
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

def get_attention_output(layer) -> Linear:
    return layer.self_attn.o_proj

def get_mlp_inputs(layer) -> list[Linear]:
    if 'phi' in layer.__class__.__name__.lower():
        return [layer.mlp.gate_up_proj]
    else:
        return [layer.mlp.gate_proj, layer.mlp.up_proj]

def get_mlp_output(layer) -> Linear:
    return layer.mlp.down_proj

def get_embeddings(model) -> list[Module]:
    return [model.model.embed_tokens]

def get_pre_head_layernorm(model) -> Module:
    return model.model.norm

def get_lm_head(model) -> Linear:
    return model.lm_head

def fuse_ln_linear(layernorm: Module, linear_layers: Iterable[Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
    
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)

def bake_mean_into_linear(linear: Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def fuse_layernorm(model):
    print("Fusing layernorm modules")
    
    #head = get_lm_head(model)
    #head.weight = Parameter(head.weight.clone())

    # We add the mean subtraction to the first embeddings
    for W in get_embeddings(model):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.model.layers

    # First we modify the layernorms to fold their weights
    for layer_adapter in layers:
        fuse_ln_linear(get_first_layernorm(layer_adapter), get_attention_inputs(layer_adapter))
        fuse_ln_linear(get_second_layernorm(layer_adapter), get_mlp_inputs(layer_adapter))

        #if model_adapter.should_bake_mean_into_linear:
            # Then we bake the mean substitution into the previous linear layers
        # bake_mean_into_linear(get_attention_output(layer_adapter))
        # bake_mean_into_linear(get_mlp_output(layer_adapter))

    fuse_ln_linear(get_pre_head_layernorm(model), [get_lm_head(model)])

    print("Fusing layernorm modules done")


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )
T = TypeVar('T')

def get_random_q(dim, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(dim, dim, dtype=torch.float64, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q

def rotate_embeddings(model, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()

def rotate_head(model, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate the head.
    W = get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in get_attention_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate output matrix of the self-attention layer.
    W = get_attention_output(layer)

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate the MLP input weights.
    for W in get_mlp_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q: torch.Tensor, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Rotate the MLP output weights and bias.
    W = get_mlp_output(layer)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def svd(W, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    U, S, Vh = torch.linalg.svd(W.to(device=device, dtype=torch.float64), full_matrices=True)
    Q = Vh.T  
    return Q

def rotate_sequential(model, device=None, final_orientation=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dtype = next(iter(model.parameters())).dtype

    # rotate embeddings
    #Q = get_random_q(model.model.embed_tokens.weight.shape[-1]).to(device)
    if 'phi' in model.__class__.__name__.lower():
        Q = svd(model.model.layers[0].self_attn.qkv_proj.weight, device)
    else:
        Q = svd(model.model.layers[0].self_attn.k_proj.weight, device)
    rotate_embeddings(model, Q, device)
    
    print("Rotating layers")
    layers = model.model.layers
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone())

        # rotate the attention inputs to match previous layer
        rotate_attention_inputs(layer, Q, device)

        # compute the Q for attention output
        #Q = get_random_q(model.model.embed_tokens.weight.shape[-1]).to(device)
        if 'phi' in model.__class__.__name__.lower():
            Q = svd(layer.mlp.gate_up_proj.weight, device)
        else:
            Q = svd(layer.mlp.gate_proj.weight, device)
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(layer.attn_shortcut_Q, Q).to(dtype=dtype)
        )
        layer.mlp_shortcut_Q = nn.Parameter(Q.T.clone())
        
        # rotate the attention output and mlp input
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the Q for the next layer
        #Q = get_random_q(model.model.embed_tokens.weight.shape[-1]).to(device)
        if idx < len(model.model.layers)-1:
            next_layer = model.model.layers[idx+1]
            if 'phi' in model.__class__.__name__.lower():
                Q = svd(next_layer.self_attn.qkv_proj.weight, device)
            else:
                Q = svd(next_layer.self_attn.k_proj.weight, device)
        layer.mlp_shortcut_Q = nn.Parameter(
            torch.matmul(layer.mlp_shortcut_Q, Q).to(dtype=dtype)
            )

        # rotate mlp output
        rotate_mlp_output(layer, Q, device)

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model, Q, device)
    
    logging.info("Rotate and slice layers done")

def tensor_transform(model, device=None):
    # replace LayerNorm with RMSNorm by fusing LayerNorm and LinearLayer
    fuse_layernorm(model)
    # rotate tensor
    rotate_sequential(model, device=device)
    # reset column-wise norms
    for layer in model.model.layers:
        if 'phi' in layer.__class__.__name__.lower():
            layer.self_attn.qkv_norm_by_column = layer.self_attn.qkv_proj.weight.norm(dim=0)
            layer.mlp.gate_up_norm_by_column = layer.mlp.gate_up_proj.weight.norm(dim=0)
        else:
            layer.self_attn.k_norm_by_column = layer.self_attn.k_proj.weight.norm(dim=0)
            layer.mlp.gate_norm_by_column = layer.mlp.gate_proj.weight.norm(dim=0)
