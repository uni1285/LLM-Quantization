import os
import sys
import fire
from pathlib import Path

import torch

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel

from transformers.trainer import L4QForceInit

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from evaluate.mmlu import EvalMMLU

os.environ["WANDB_DISABLED"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate(
    # model/data params
    base_model: str = "llama-2-7b",
    peft_model: str= "output/l4q",
    # evaluate MMLU ?
    eval_mmlu: bool = True,
    ntrain: int = None,
    init: bool = False,
):
    base_model = os.path.join("/SHARE_ST/vlsi/anonymous/data/", base_model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, peft_model)
    if init is True:
        qerror = L4QForceInit(model, "default", 0)
        print("#####################\nQuant error profiling result: ", qerror)
        qerror = L4QForceInit(model, "default", 1)
        print("#####################\nScaling error profiling result: ", qerror)
        import pdb; pdb.set_trace()
    model = model.merge_and_unload()
    model.to(torch.float)
    model.eval()

    with torch.no_grad():
        if eval_mmlu:
            if ntrain is not None:
                EvalMMLU(device="cuda", ntrain=ntrain, data_dir="evaluate/data", model=model, tokenizer=tokenizer, save_dir=peft_model)
            else:
                EvalMMLU(device="cuda", ntrain=0, data_dir="evaluate/data", model=model, tokenizer=tokenizer, save_dir=peft_model)
                EvalMMLU(device="cuda", ntrain=5, data_dir="evaluate/data", model=model, tokenizer=tokenizer, save_dir=peft_model)


if __name__ == "__main__":
    fire.Fire(evaluate)
