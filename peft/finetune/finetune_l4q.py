import os
import sys
import fire
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    get_peft_model,
    set_peft_model_state_dict,
    PeftModel,
)

from transformers.trainer import L4QTrainer, L4QProfiler, enable_lora
from peft.tuners.lora.config import L4QConfig

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from utils.prompter import Prompter

os.environ["WANDB_DISABLED"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(
    # model/data params
    base_model: str = "llama-2-7b",
    data_path: str = "data/alpaca_data_cleaned_archive.json",
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    output_dir: str = "output/l4q",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 0.505,  # 400 steps
    learning_rate: float = 2e-5,
    cutoff_len: int = 2048,
    val_set_size: int = 1000,
    resume_from_checkpoint: str = None,
    warmup_steps: int = 50,
    logging_steps: int = 5,
    eval_steps: int = 100,
    save_steps: int = 100,
    # lora hyperparams
    lora_r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    # L4Q hyperparameters
    qerror: bool = False,
    bf: bool = True,
    w_bits: int = 4,
    g_size: int = 128,
    l4q_start_steps: int = -1,
):
    seed = 42

    output_dir = os.path.join(os.getcwd(), output_dir, base_model, str(w_bits) + "bit")
    base_model = os.path.join("/SHARE_ST/vlsi/anonymous/data/", base_model)
    os.makedirs(output_dir, exist_ok=True)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"prompt template: {prompt_template_name}\n"
            f"output_dir: {output_dir}\n"

            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"

            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"

            f"w_bits: {w_bits}\n"
        )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    config = L4QConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        l4q=False if l4q_start_steps > 0 else True,
        w_bits=w_bits,
        g_size=g_size,
    )
    # model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    model.config.use_cache = False
    if bf is True:
        model.to(torch.bfloat16)

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        enable_lora(model)
        seed += 1

    model.print_trainable_parameters()

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        prompter = Prompter(prompt_template_name)
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = L4QTrainer(
        model=model,
        l4q_start_steps=l4q_start_steps,
        args=TrainingArguments(
            label_names=['labels'],
            num_train_epochs=num_epochs,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            learning_rate=learning_rate,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            bf16=bf,
            output_dir=output_dir,
            # save_total_limit=4,
            load_best_model_at_end=True if val_set_size > 0 else False,
            overwrite_output_dir=True,
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.peft_config['default'].l4q = True
    model.peft_config['default'].lora_dropout = 0

    if qerror is True:
        qerror = L4QProfiler(model, "default", 0)
        print("#####################\nQuant error profiling result: ", qerror)
        qerror = L4QProfiler(model, "default", 1)
        print("####################\nClipping error profiling result: ", qerror)
        qerror = L4QProfiler(model, "default", 2)
        print("####################\nStep size ratio profiling result: ", qerror[0] / qerror[3], qerror[1] / qerror[2], qerror[2] / qerror[3], 1.0)
        exit(0)

    train_result = trainer.train()
    trainer.save_model(output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
