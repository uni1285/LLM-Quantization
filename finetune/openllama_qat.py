import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(wd))

import lightning as L
import torch
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from functools import partial

from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

from generate.base import generate
from lit_gpt.qat import GPT, Block, Config, enable_lsq_lora
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt

version = "v1.3.2"  # quant bias

# Hyperparameters
batch_size = 128
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
# max_iters = 51200  # 400 step  # around train dataset size
# save_interval = 200
eval_interval = 50
eval_iters = 100  # will be 2000 in the last evaluation
eval_max_new_tokens = 100
log_interval = 1

lora_r = 0
lora_alpha = 0
lora_dropout = 0.0
lora_query = False
lora_key = False
lora_value = False
lora_projection = False
lora_mlp = False
# lora_head = True

# warmup_steps = 20
# learning_rate = 1e-3
# weight_decay = 0.01
# lsq_start = 50
# w_bits = 4
# q_granul = "group"
# gs = 128
# precision = "bf16-mixed"

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca_openllama/"),
    out_dir: Path = Path("out/l4q/openllama"),
    devices: Optional[int] = 1,
    precision: Optional[str] = "bf16-true",
    max_iters: Optional[int] = 25600,
    save_interval: Optional[int] = 100,
    warmup_steps: Optional[int] = 20,
    learning_rate: Optional[float] = 1e-4,
    weight_decay: Optional[float] = 0.01,
    # LSQ-LoRA Parameters
    lora_head: Optional[bool] = False,
    lm_head_quant: Optional[bool] = False,
    qb_train: Optional[bool] = True,
    lsq_start: Optional[int] = 0,
    w_bits: Optional[int] = 4,
    q_granul: Optional[str] = "group",
    gs: Optional[int] = 64,
) -> None:

    out_dir = out_dir / "openllama" / (str(w_bits) + 'bits')
    precision = precision or get_default_supported_precision(training=True)
    checkpoint_dir = Path("/home/anonymous/works/llm/alpaca-lora/open_llama_3b")
    plugins = None
    strategy = "auto"

    logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, devices,
                  max_iters, save_interval, warmup_steps, learning_rate, weight_decay,
                  lora_head, lm_head_quant, qb_train, lsq_start, w_bits, q_granul, gs)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, devices: int,
         max_iters: int, save_interval: int, warmup_steps: int, learning_rate: float, weight_decay: float,
         lora_head: bool, lm_head_quant: bool, qb_train: bool, lsq_start: int, w_bits: int, q_granul: str, gs: int) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_name(
        name="open_llama_3b",
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
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)
    model.lm_head_quant = lm_head_quant
    model.qb_train = qb_train
    # mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # max_iters // batch_size)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    # print(model)
    # print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, scheduler, train_data, val_data, checkpoint_dir, out_dir,
          max_iters, save_interval, warmup_steps, learning_rate, lsq_start)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    # save_path = out_dir / "lit_model_lora_finetuned.pth"
    # save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    max_iters: int,
    save_interval: int,
    warmup_steps: int,
    learning_rate: float,
    lsq_start: int,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    # validate(fabric, model, val_data, tokenizer, max_iters=2)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    lsq_started = False
    total_t0 = time.perf_counter()

    for iter_num in range(1, max_iters + 1):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if step_count >= lsq_start and not lsq_started:
            model.config.lsq = True
            enable_lsq_lora(model)
            lsq_started = True
            fabric.print(f"LSQ train start with {model.config.__dict__}")
            fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
            fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, train_data, longest_seq_ix if iter_num == 1 else None)

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            if step_count > warmup_steps:
                scheduler.step()
            step_count += 1

        total_lengths += input_ids.numel()
        if not is_accumulating and step_count % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * micro_batch_size, lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num:>5d}  step {step_count:>4d}  loss {loss_item:.4f}  "
                f"memory {torch.cuda.max_memory_allocated() / 1e9:.02f} GB  iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, max_iters=eval_iters)
            t1 = time.perf_counter() - t0
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1:.2f}s")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)

        torch.cuda.empty_cache()


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(max_iters)
    for k in range(max_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8)
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


def fsdp_auto_wrap_policy(block: torch.nn.Module):

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={block})

    auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
