import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch: left shape {left.shape} does not match "
            f"right shape {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params["wpe"])
    gpt.tok_embedding.weight = assign(gpt.tok_embedding.weight, params["wte"])

    for block_index, block_params in enumerate(params["blocks"]):
        block = gpt.transformer_blocks[block_index]
        q_w, k_w, v_w = np.split(
            block_params["attn"]["c_attn"]["w"], 3, axis=-1
        )
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            block_params["attn"]["c_attn"]["b"], 3, axis=-1
        )
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight,
            block_params["attn"]["c_proj"]["w"].T,
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias,
            block_params["attn"]["c_proj"]["b"],
        )

        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight,
            block_params["mlp"]["c_fc"]["w"].T,
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias,
            block_params["mlp"]["c_fc"]["b"],
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight,
            block_params["mlp"]["c_proj"]["w"].T,
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias,
            block_params["mlp"]["c_proj"]["b"],
        )

        block.norm1.scale = assign(block.norm1.scale, block_params["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, block_params["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, block_params["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, block_params["ln_2"]["b"])

    gpt.final_layer_norm.scale = assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = assign(gpt.final_layer_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
