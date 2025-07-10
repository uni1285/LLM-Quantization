import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from datasets import load_dataset

from tqdm import tqdm
import time
import argparse


@torch.no_grad()
def EvalPPL(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    model.seqlen = 2048

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="PPL layer modification"):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    model.config.use_cache = use_cache

    return ppl.item()


def eval_wikitext2(args):
    model = LlamaForCausalLM.from_pretrained(args.model_dir).to("cpu")
    if args.peft:
        model = PeftModel.from_pretrained(model, args.peft).model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testloader = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    eval_ppl = EvalPPL(model, testloader, args.device, "wikitext2")


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="wikitext2", help="ppl task")
    parser.add_argument("--model_dir", type=str, help="huggingface model directory")
    parser.add_argument("--device", type=str, default="cuda", help="target device")
    parser.add_argument("--peft", type=str, help="adapter directory")

    return parser.parse_args()



if __name__ == "__main__":
    args = parsing_args()
    if args.peft:
        print(f"Task: {args.task} | Model: {args.model_dir} | Adapter: {args.peft}")
    else:
        print(f"Task: {args.task} | Model: {args.model_dir}")

    if args.task=="wikitext2":
        eval_wikitext2(args)
