import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import transformer_lens.utils as utils
import seaborn as sns


def imshow_px(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line_px(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter_px(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def imshow(
    tensor,
    tokens=None,
    title=None,
    legend=False,
    colorbar=True,
    cmap='RdBu',
    norm=None,
    figsize=(8, 6),
    colorbar_height=1,
    num_in_block=True,
    xlabel="Layer",
    ylabel="Token",
    centered_colorbar=False,
):
    
    plt.figure(figsize=figsize)
    plt.imshow(utils.to_numpy(tensor), cmap=cmap, norm=norm)
    if tokens is not None:
        plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    if colorbar:
        if centered_colorbar:
            max_val = np.abs(tensor.clone().cpu()).max()
            vmin = -max_val
            vmax = max_val
        else:
            vmin = tensor.min()
            vmax = tensor.max()
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color='gray')
        plt.imshow(utils.to_numpy(tensor), cmap=cmap, norm=norm)
        plt.colorbar(ticks=[vmin, 0, vmax], format='%.4f')
        plt.clim(vmin, vmax)
    if num_in_block:
        pass
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def line(
    list
    ,xlabel="Layer"
    ,ylabel="Prob"
    ,title=None
    ,vline=True
    ,legend=True
):
    plt.plot(range(len(list)), list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend
    plt.title(title)
    if vline:
        for i in range(len(list)):
            plt.axvline(x=i, color='gray', linestyle='--')
    if legend:
        plt.legend()
    plt.show()


def hidden_states_projection(
    cache,
    tokens,
    str_tokens,
    use_cache=False,
    is_without_bos=0,
    num_layers=32,
    cmap="Reds",
    norm=None,
    unembed_weight=None,
    tokenizer=None,
):
    # projection without decomposition, using only resid_mid
    sequence_length = len(tokens[0])
    num_components = sequence_length

    if not use_cache:
        h_matrix_resid = np.empty((sequence_length, num_layers), dtype=object)
        for layer in tqdm.tqdm(range(num_layers)):
            
            for i in range(num_components):

                i_logits = torch.matmul(norm(cache[f"blocks.{layer}.hook_resid_post"][i].float()), unembed_weight)
                max_prob = max(torch.softmax(i_logits, dim=-1))
                meaning = tokenizer.decode(torch.softmax(i_logits, dim=-1).argmax(dim=-1))
                
                h_matrix_resid[i, layer] = {"prob": max_prob, "meaning": meaning, "logits": i_logits}

        prob_matrix_resid = np.zeros((sequence_length, num_layers))
        for i in range(sequence_length):
            for j in range(num_layers):
                prob_matrix_resid[i, j] = h_matrix_resid[i, j]["prob"]
    else:
        pass

    generated_tokens = str_tokens[is_without_bos:]  # get rid of <s>
    generated_tokens = [token.replace("Ġ", "") for token in generated_tokens]
    print(generated_tokens)

    # decomposition projection
    plt.figure(figsize=(30, 10))
    plt.xlabel("Layer")
    plt.ylabel("Tokens")
    sns.heatmap(prob_matrix_resid[is_without_bos:, :], cmap=cmap)
    for i in range(is_without_bos, num_components):
        for j in range(num_layers):
            plt.text(j + 0.5, i + 0.5 - is_without_bos, f"{h_matrix_resid[i, j]['meaning']}", ha="center", va="center", color="black")
    plt.yticks(np.arange(is_without_bos, num_components), generated_tokens, rotation=20)
    plt.show()

    return h_matrix_resid, prob_matrix_resid


def get_prediction_per_layer(
    model,
    cache,
    logits=None,
    layer_id=-1,
    norm=None,
    unembed_weight=None,
    tokenizer=None,
):
    # get the predicted token
    if logits is not None:
        pred_token = logits[0, :].argmax(dim=-1)[layer_id]
        print("model output token: ",pred_token)
        print("model output str: ", tokenizer.decode(logits[0, layer_id, :].argmax(dim=-1)))

    # iterate over layers
    for layer_idx in range(model.cfg.n_layers):
        last_token_h = cache[f"blocks.{layer_idx}.hook_resid_post"][layer_id]
        assert len(last_token_h) == model.cfg.d_model

        last_token_h = norm(last_token_h)

        #TODO: interesting scene to see if we apply layer norm to the last layer
        #BUG: tl package rmsnorm not * weight
        if layer_idx == model.cfg.n_layers - 1:
            last_token_h = cache["ln_final.hook_normalized"][layer_id].to(model.cfg.dtype)

        # upproject to vocab size using unembedding weight
        last_token_proj = torch.matmul(last_token_h.bfloat16(), unembed_weight.bfloat16())

        last_token_prob = torch.softmax(last_token_proj, dim=-1)
        last_token_meaning = tokenizer.decode(last_token_proj.argmax(dim=-1))
        max_prob = last_token_prob.max().item()
        
        target_prob = last_token_prob[pred_token].item()
        target_rank = (last_token_prob > target_prob).sum()
        
        print(f"Layer {layer_idx} | output prob: {max_prob:.4f} | output meaning: {last_token_meaning} | pred prob: {target_prob:.4f} | rank: {target_rank}")