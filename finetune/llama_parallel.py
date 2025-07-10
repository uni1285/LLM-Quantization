"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import sys
import time
import random
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(wd))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Dataset

from generate.llama import generate
from lit_llama.lora import LLaMA, Config, mark_only_lora_as_trainable, enable_lsq_lora, lora_state_dict, merge_lsq_lora_weights
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lit_gpt.utils import num_parameters

import deepspeed
from deepspeed.pipe import PipelineModule

version = "v1.3.2"

eval_interval = 50
save_interval = 200
eval_iters = 100
log_interval = 1

# Hyperparameters
batch_size = 128
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 25600
# max_iters = 204800

lora_r = 4
lora_alpha = 8
lora_dropout = 0.05
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True
lora_head = True

# lsq_start = 50
# w_bits = 4
# q_granul = "group"
# gs = 128


def main(
    local_rank: int = 0,
    data_dir: str = "data/alpaca_llama-1",
    out_dir: Path = Path("out/llsq/llama-1/old"),
    model_size: Optional[str] = "7b",
    devices: Optional[int] = 1,
    precision: Optional[str] = "bf16-true",
    # Training Parameters
    warmup_iters: Optional[int] = 20,
    learning_rate: Optional[float] = 1e-4,
    weight_decay: Optional[float] = 0.01,
    # LSQ-LoRA Parameters
    lsq_start: Optional[int] = 10,
    w_bits: Optional[int] = 4,
    q_granul: Optional[str] = "group",
    gs: Optional[int] = 128,
    # Evaluate
    lm_evaluate: Optional[bool] = True,
    eval_tasks: Optional[List[str]] = ['hellaswag', 'piqa', 'arc_easy','arc_challenge', 'winogrande', 'boolq', 'openbookqa'],
    num_fewshot: int = 0,
    bootstrap_iters: int = 2,
    temperature: float = 1.0,
    save_filepath: Path = None,
):
    seed = 1337
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deepspeed.runtime.utils.set_random_seed(seed)

    deepspeed.init_distributed(dist_backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    pretrained_path = "/SHARE_ST/vlsi/anonymous/data/llama-" + model_size + "/lit_model.pth"
    tokenizer_path = "/SHARE_ST/vlsi/anonymous/data/llama-" + model_size + "/tokenizer.model"

    out_dir = out_dir / model_size / (str(w_bits) + 'bits')
    os.makedirs(out_dir, exist_ok=True)

    config = Config.from_name(
        name=model_size.upper(),
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

    # print(f"Setting model {str(pretrained_path)!r} with {config.__dict__}")

    model = LLaMA(config)
    mark_only_lora_as_trainable(model)
    print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    train_data, val_data = load_datasets(data_dir=data_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    # print(
        # f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        # f" {model.max_seq_length} and context length is {model.config.block_size}"
    # )
    train_set = get_loader(train_data, local_rank)
    val_set = get_loader(val_data, local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters // batch_size)

    model = prepare_parallel(model, devices)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_set,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config={
            "bfloat16_enabled" : True,
            "train_batch_size" : 128,
            "train_micro_batch_size_per_gpu" : 1,
            "steps_per_print" : 1,
        })
    engine.load_checkpoint(load_dir=pretrained_path, load_module_strict=False)
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    train(engine, model, val_set, tokenizer_path, out_dir, lsq_start)


def train(
    engine,
    model: torch.nn.Module,
    val_set: Dataset,
    tokenizer_path: str,
    out_dir: Path,
    lsq_start: int,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    lsq_started = False

    # validate(model, val_data, tokenizer_path)  # Sanity check

    for iter_num in range(1, max_iters+1):
        if step_count >= lsq_start and not lsq_started:
            enable_lsq_lora(model)
            lsq_started = True
            print(f"LSQ train start")
            print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
            print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

        t0 = time.time()
        engine.train_batch()
        # logits = engine(input_ids)
        # loss = loss_fn(logits, targets)
        # engine.backward(loss)
        # engine.step()

        if iter_num % gradient_accumulation_iters:
            step_count += 1

        if step_count % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.time()
            print(
            f"iter {iter_num:>5d}  step {step_count:>4d}  loss {loss_item:.4f}  "
            f"memory {torch.cuda.max_memory_allocated() / 1e9:.02f} GB  iter time: "
            f"{(t1 - t0) * 1000:.2f}ms (optimizer.step)")

        # if step_count % eval_interval == 0:
            # val_loss = validate(model, val_data, tokenizer_path)
            # print(f"step {iter_num}: val loss {val_loss:.4f}")
            # dist.barrier()

        if step_count % save_interval == 0:
            print(f"Saving LoRA weights to {out_dir}")
            checkpoint = lora_state_dict(model)
            engine.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

    torch.cuda.empty_cache()


def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_new_tokens=100,
    )
    model.reset_cache()
    output = tokenizer.decode(output)
    return output


@torch.no_grad()
def validate(model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    print("Validating ...")
    # model.eval()
    # losses = torch.zeros(eval_iters)
    # for k in range(eval_iters):
        # input_ids, targets = get_batch(val_data)
        # logits = model(input_ids)
        # loss = loss_fn(logits, targets)
        # losses[k] = loss.item()
    # out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."

    output = generate_response(model, instruction, tokenizer_path)
    print(instruction)
    print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_loader(data: list, local_rank: int):
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    ix = torch.randperm(len(data))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if local_rank == 0:
        dist.barrier()

    return dataset(x, y)


class dataset(Dataset):
    def __init__(self,x, y):
        self.x = x
        self.y = y
    def __getitem__(self,i):
        return self.x[i],self.y[i]
    def __len__(self):
        return len(self.x)


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def prepare_parallel(model: LLaMA, devices: int):
    n_layer = model.config.n_layer
    p_layer = n_layer // devices

    module_list = [model.transformer.wte.to(0)]
    for i in range(n_layer):
        block = model.transformer.h[i]
        module_list.append(block.to(i // p_layer))
    module_list.append(model.lm_head.to(devices-1))

    net = PipelineModule(layers=module_list,
                         loss_fn=loss_fn,
                         num_stages=2)

    return net


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
