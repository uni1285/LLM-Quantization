import os
import argparse
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from pyhessian import hessian
from density_plot import get_esd_plot

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, set_peft_model_state_dict
from datasets import load_dataset

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

class HessianDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx][:-1]), torch.tensor(self.targets[idx][1:])


# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


def loss_landscape(args):
    # prepare model
    print("Model preparation")
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'pad_token': "</s>"})

    model = LlamaForCausalLM.from_pretrained(args.model, device_map="cpu")
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    elif args.qalora:
        qalora = torch.load(args.qalora)
        set_peft_model_state_dict(model, qalora)

    for name, param in model.named_parameters():
        # if "lora_A" in name or "lora_B" in name:
            # param.requires_grad = True
        param.requires_grad = True

    model.eval()
    model = model.cuda()

    prompter = Prompter("alpaca")
    def tokenize(prompt, max_length=args.max_length):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,  # "max_length",
            # return_tensors=None,
        )
        # if (
            # result["input_ids"][-1] != tokenizer.eos_token_id
            # and len(result["input_ids"]) < 1304
            # and add_eos_token
        # ):
            # result["input_ids"].append(tokenizer.eos_token_id)
            # result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    # prepare data
    print("Data preparation")
    data = load_dataset("json", data_files=args.data)
    dataset = data["train"].train_test_split(test_size=args.test_size, shuffle=True, seed=42)
    data_val = (dataset["test"].shuffle().map(generate_and_tokenize_prompt))
    dataset_val = HessianDataset(data_val['input_ids'], data_val['labels'])
    dataloader_val = DataLoader(dataset_val, batch_size=1)

    criterion = torch.nn.CrossEntropyLoss()

    print("Hessian preparation")
    hessian_comp = hessian(model=model, criterion=criterion, dataloader=dataloader_val, cuda=True)
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")

    print("Eigenvalue preparation")
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")

    lamd_1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    lamd_2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)

    model_pertb_1 = model
    model_pertb_2 = model

    loss_list = []

    print("Loss Landscape figuration")
    with torch.no_grad():
        for l1 in tqdm(lamd_1, leave=True):
            for l2 in tqdm(lamd_2, leave=False):
                model_pertb_1 = get_params(model, model_pertb_1, top_eigenvector[0], l1)
                model_pertb_2 = get_params(model_pertb_1, model_pertb_2, top_eigenvector[1], l2)
                loss = 0.
                for inputs, targets in dataloader_val:
                    logits = model_pertb_2(inputs.cuda()).logits
                    outputs = logits.reshape(-1, logits.size(-1))
                    loss += criterion(outputs, targets.reshape(-1).cuda()).item()
                loss_list.append((l1, l2, loss))
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")

    loss_list = np.array(loss_list)

    fig = plt.figure()
    landscape = fig.add_subplot(projection="3d")
    landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2], alpha=0.8, cmap='viridis')
    landscape.set_title('Loss landscape')
    landscape.set_xlabel('lambda_1')
    landscape.set_ylabel('lambda_2')
    landscape.set_zlabel('Loss')

    landscape.dist = 6
    plt.savefig('landscape/landscape.png')



def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="alpaca datset directory",
                        default="/home/anonymous/works/llm/alpaca-lora/data/alpaca_data_cleaned_archive.json")
    parser.add_argument("--model", type=str, help="huggingface model directory")
    parser.add_argument("--lora", type=str, help="LoRA directory")
    parser.add_argument("--qalora", type=str, help="QA-LoRA directory")
    parser.add_argument("--test_size", type=int, help="QA-LoRA directory",
                        default=1)
    parser.add_argument("--max_length", type=int, help="Model cutoff length",
                        default=128)

    return parser.parse_args()


if __name__ == "__main__":
    args = parsing_args()
    if args.lora:
        print(f"Model: {args.model} | LoRA: {args.lora}")
    elif args.qalora:
        print(f"Model: {args.model} | QA-LoRA: {args.qalora}")
    else:
        print(f"Model: {args.model}")

    loss_landscape(args)
