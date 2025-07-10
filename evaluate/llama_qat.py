import sys
import json
import time
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.tokenizer import Tokenizer
from lit_llama.qat import LLaMA, Block, Config, merge_lsq_lora_weights
from lit_gpt.utils import load_checkpoint

from evaluate.lm_eval_harness import EvalHarnessBase
from evaluate.eval_mmlu import EvalMMLU
from evaluate.eval_ppl import EvalPPLv1
from evaluate.utils import dt_analysis, loss_landscape

from datasets import load_dataset

lora_r = 0
lora_alpha = 0
lora_dropout = 0.0
lora_query = False
lora_key = False
lora_value = False
lora_projection = False
lora_mlp = False
# lora_head = True
lsq = False
q_granul = 'group'
gs = 128


def main(
    input: str = "",
    lora_path: Path = Path("out/model.pth"),
    model_size: str = "7b",
    w_bits: int = 4,
    temperature: float = 1.0,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = "32-true",
    lora_head: Optional[bool] = False,
    lm_head_quant: Optional[bool] = False,

    analysis: bool = False,
    layers: Optional[List[int]] = [-1],
    verbose: bool = False,
    landscape: str = None,  # "data/alpaca_llama-1_256",
    test_size: int = 64,
    seed: int = 42,
    mmlu: bool = True,
    eval_tasks: Optional[List[str]] = ['hellaswag', 'piqa', 'arc_easy','arc_challenge', 'winogrande', 'boolq', 'openbookqa'],
    batch_size: int = 1,
    num_fewshot: int = 0,
    bootstrap_iters: int = 2,
    save_filepath: Optional[Path] = None,
    ppl: bool = False,
    device_: Optional[str] = "cuda"
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    if model_size == "30b":
        precision = "bf16-true"
    fabric = L.Fabric(accelerator=device_, devices=1, precision=precision)
    fabric.launch()

    checkpoint_dir = Path("/SHARE_ST/vlsi/anonymous/data/llama-" + model_size)
    model_file = "lit_model.pth"
    tokenizer_file = "tokenizer.model"
    checkpoint_path = checkpoint_dir / model_file
    tokenizer_path = checkpoint_dir / tokenizer_file
    tokenizer = Tokenizer(tokenizer_path)

    config = Config.from_name(
        name=model_size.upper(),
        block_size=2048,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
        lsq=False,
        w_bits=w_bits,
        q_granul=q_granul,
        gs=gs,
    )
    fabric.print(f"Loading model {str(lora_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = LLaMA(config)
    model.lm_head_quant = lm_head_quant
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, lora_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    if analysis:
        print("Model parameter delta analysis")
        if device_ == "cuda":
            model.to("cpu")
        print("Model offload to cpu")
        delta = dt_analysis(model, verbose, layers)
        if device_ == "cuda":
            model.to("cuda")
        print("Model back to cuda")

    if landscape:
        data_dir = Path(landscape)
        val_data = torch.load(data_dir / "test.pt")[:test_size]
        loss_landscape(model=model, data=val_data, seed=seed, model_size=model_size, w_bits=w_bits, k=5)

    merge_lsq_lora_weights(model)

    if mmlu:
        EvalMMLU(device=fabric.device, ntrain=0, data_dir=None, model=model, tokenizer=tokenizer, save_dir=None)
        EvalMMLU(device=fabric.device, ntrain=5, data_dir=None, model=model, tokenizer=tokenizer, save_dir=None)

    if len(eval_tasks) > 0:
        eval_harness = EvalHarnessBase(fabric, model, tokenizer, batch_size, temperature)
        results = eval_harness.run_eval(
            eval_tasks=eval_tasks, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters, use_cache=False)
        if save_filepath is None:
            print(results)
        else:
            print(f"Saving results to {str(save_filepath)!r}")
            data = json.dumps(results)
            with open(save_filepath, "w") as fw:
                fw.write(data)

    if ppl:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testloader = tokenizer.encode("\n\n".join(test["text"]), device=fabric.device)  # return_tensors="pt")
        eval_ppl = EvalPPLv1(model=model, testenc=testloader, dev=fabric.device, dataset="wikitext2")


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
