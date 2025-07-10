import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset

from tqdm import tqdm
import time
import argparse


@torch.no_grad()
def EvalPPLv2(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    model.seqlen = 2048

    testenc = testenc[None,:].to(torch.int64)
    nsamples = testenc.numel() // model.seqlen

    # layers = model.model.layers
    layers = model._forward_module.transformer.h
    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model._forward_module.transformer.wte = model._forward_module.transformer.wte.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.n_embd), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, cos, sin, mask, input_pos):
            inps[cache["i"]] = x
            cache["i"] += 1
            cache["attention_mask"] = mask
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    model._forward_module.transformer.wte = model._forward_module.transformer.wte.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="PPL layer modification"):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            T = inps[j].size(0)
            cos = model._forward_module.cos[:T]
            sin = model._forward_module.sin[:T]
            outs[j] = layer(x=inps[j].unsqueeze(0), cos=cos, sin=sin, mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # if model.model.norm is not None:
        # model.model.norm = model.model.norm.to(dev)
    if model._forward_module.transformer.ln_f is not None:
        model._forward_module.transformer.ln_f = model._forward_module.transformer.ln_f.to(dev)
    model._forward_module.lm_head = model._forward_module.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model._forward_module.transformer.ln_f is not None:
            hidden_states = model._forward_module.transformer.ln_f(hidden_states)
        lm_logits = model._forward_module.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).to(torch.float16), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    return ppl.item()


@torch.no_grad()
def EvalPPLv1(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    model.seqlen = 2048

    testenc = testenc[None,:].to(torch.int64)
    nsamples = testenc.numel() // model.seqlen

    # layers = model.model.layers
    layers = model._forward_module.transformer.h
    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model._forward_module.transformer.wte = model._forward_module.transformer.wte.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.n_embd), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, rope, mask, seqlen):
            inps[cache["i"]] = x
            cache["i"] += 1
            cache["attention_mask"] = mask
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    model._forward_module.transformer.wte = model._forward_module.transformer.wte.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="PPL layer modification"):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            T = inps[j].size(0)
            rope = model._forward_module.rope_cache[:T]
            outs[j] = layer(x=inps[j].unsqueeze(0), rope=rope, mask=attention_mask, max_seq_length=model.seqlen)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # if model.model.norm is not None:
        # model.model.norm = model.model.norm.to(dev)
    if model._forward_module.transformer.ln_f is not None:
        model._forward_module.transformer.ln_f = model._forward_module.transformer.ln_f.to(dev)
    model._forward_module.lm_head = model._forward_module.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model._forward_module.transformer.ln_f is not None:
            hidden_states = model._forward_module.transformer.ln_f(hidden_states)
        lm_logits = model._forward_module.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).to(torch.float16), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    return ppl.item()

