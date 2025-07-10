import os
from tqdm import tqdm
from typing import Union, Optional, List
from pathlib import Path
from functools import reduce

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lit_gpt import GPT
from lit_llama import LLaMA
from lit_gpt.lora import LoRALinear, LoRAQKVLinear

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from pyhessian import hessian



def dt_analysis(model: Union[GPT, LLaMA], verbose: Optional[bool] = False, layers: Optional[List[int]] = [-1]):
    if -1 in layers:
        layers = list(range(100))
    delta = {}
    for name, module in model.named_modules():
        if name == "lm_head":
            pass
        elif isinstance(module, LoRAQKVLinear):
            m = name.split('.')
            layer = m[2]
            target = m[-2]+'.'+m[-1]
            if layer not in delta:
                print("Layer: ", layer)
                delta[layer] = {}
            if int(layer) in layers:
                weight = module.linear.weight.view(3, -1, module.linear.weight.shape[-1])
                lora = (module.conv1d(
                        module.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                        module.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                    ).squeeze(0) * module.l_scale).view(3, -1, weight.shape[-1])
                lora_absm = lora.abs().mean(dim=-1)
                weight_sum = weight.data + lora.data
                delta[layer]['w_total'] += (weight_new.mean(dim=-1) - weight.mean(dim=-1)).abs().sum().data
                names = ["q_proj", "k_proj", "v_proj"]
                for i in range(3):
                    ws = weight_sum[i].flatten().detach().numpy()
                    wm = 0.0
                    om, gm = 0, 0
                    for o in range(weight.shape[1]):
                        for g in range(weight.shape[-1]//128):
                            wg = weight[i, o, g*128 : (g+1)*128].flatten().detach().numpy()
                            wsg = weight_sum[i, o, g*128 : (g+1)*128].flatten().detach().numpy()
                            if abs((wsg - wg).mean()) > wm:
                                wm = abs((wsg - wg).mean())
                                om = o
                                gm = g
                    wg = weight[i, om, gm*128 : (gm+1)*128].flatten().detach().numpy()
                    wsg = weight_sum[i, om, gm*128 : (gm+1)*128].flatten().detach().numpy()
                    print(f"{weight_sum[i].mean().data:.2e} {lora[i].mean().abs().data/weight[i].abs().mean().data:.2e} {wsg.mean():.2e} {wm/abs(wg).mean().data:.2e}")
        elif isinstance(module, LoRALinear):
            m = name.split('.')
            layer = m[2]
            target = m[-2]+'.'+m[-1]
            if layer not in delta:
                delta[layer] = {}
            if int(layer) in layers:
                weight = module.linear.weight
                lora = module.lora_B @ module.lora_A * module.l_scale
                lora_absm = lora.abs().mean()
                weight_sum = weight.data + lora.data
                w = w[abs(w) < 6*w.std()]
                ws = weight_sum.flatten().detach().numpy()
                wm = 0.0
                om, gm = 0, 0
                for o in range(weight.shape[0]):
                    for g in range(weight.shape[-1]//128):
                        wg = weight[o, g*128 : (g+1)*128].flatten().detach().numpy()
                        wsg = weight_sum[o, g*128 : (g+1)*128].flatten().detach().numpy()
                        if abs((wsg - wg).mean()) > wm:
                            wm = abs((wsg - wg).mean())
                            om = o
                            gm = g
                wg = weight[om, gm*128 : (gm+1)*128].flatten().detach().numpy()
                wsg = weight_sum[om, gm*128 : (gm+1)*128].flatten().detach().numpy()
                print(f"{weight_sum.mean():.2e} {lora.mean().abs().data/weight.abs().mean().data:.2e} {wsg.mean():.2e} {wm/abs(wg).mean().data:.2e}")
    if verbose:
        delta_s = sorted(delta.items(), key=lambda x:x[1]['w_total'])
        print("weight bias sorted")
        for i in range(len(delta_s)):
            print(f"{delta_s[i][0]} {delta_s[i][1]['w_total'] / num * 32}")
        delta_s = sorted(delta.items(), key=lambda x:x[1]['s_total'])
        print("quant scale sorted")
        for i in range(len(delta_s)):
            print(f"{delta_s[i][0]} {delta_s[i][1]['s_total'] / num * 32}")
    return delta


def get_module(model, name):
    names = name.split(sep='.')
    return reduce(getattr, names, model)

def cos_similarity(target: Union[GPT, LLaMA], model: Union[GPT, LLaMA]):
    for name, module in model.named_modules():
        cos = {str(l): None for l in range(32)}
        if isinstance(module, LoRALinear)
            import pdb; pdb.set_trace()
            if "lm_head" in name:
                pass
            else:
                m = name.split('.')
                l_num = m[2]
                l_name = m[-2]+'.'+m[-1]
                elif isinstance(module, LoRAQKVLinear):
                    t_module = get_module(target, name)
                    for i in range(3):
                        dim = module.linear.weight.shape()[0] / 3
                        cos[l_num][l_name] = nn.CosineSimilarity(t_module.linear.weight[int(i*dim) : int((i+1)*dim)].numel(),
                                                                 module.linear[int(i*dim) : int((i+1)*dim)].weight.numel())
                else:
                    t_module = get_module(target, name)
                    cos[l_num][l_name] = nn.CosineSimilarity(t_module.linear.weight.numel(), module.linear.weight.numel())



def set_landscape(model: Union[GPT, LLaMA], value: bool) -> None:
    for block in model.transformer.h:
        block.attn.landscape = value


class HessianDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx][:-1], device="cuda"), torch.tensor(self.targets[idx][1:]).type(torch.LongTensor).to("cuda")


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip([p for p in model_orig.parameters() if p.requires_grad], [p for p in model_perb.parameters() if p.requires_grad], direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


def loss_landscape(model: Union[GPT, LLaMA], data, seed, model_size, w_bits, k):
    if isinstance(model, LLaMA):
        out_dir = Path("landscape/l4q/llama-1") / model_size / (str(w_bits) + 'bits')
    elif isinstance(model, GPT):
        out_dir = Path("landscape/l4q/llama-2") / model_size / (str(w_bits) + 'bits')
    os.makedirs(out_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if not 'linear.weight' in name:
            if 'lora_lsq' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            if 'lm_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    dataset_val = HessianDataset([d['input_ids'] for d in data], [d['labels'] for d in data])
    dataloader_val = DataLoader(dataset_val, batch_size=1)
    criterion = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(k), leave=False, desc="seed"):
        set_landscape(model, True)  # enable 2nd order gradient to computed hessian
        print("Hessian preparation")
        hessian_comp = hessian(model=model, criterion=criterion, dataloader=dataloader_val, cuda=True, seed=seed)
        # print(f"Memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")
        print("Eigenvalue preparation")
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        # print(f"Memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")
        lamd_1 = np.linspace(-0.5, 0.5, 17).astype(np.float32)
        lamd_2 = np.linspace(-0.5, 0.5, 17).astype(np.float32)
        model_pertb_1 = model
        model_pertb_2 = model
        loss_list = []
        set_landscape(model, False)  # enable fast inference
        print("Loss Landscape figuration")
        with torch.no_grad():
            for l1 in tqdm(lamd_1, leave=True, desc="Lambda 1"):
                for l2 in tqdm(lamd_2, leave=False, desc="Lambda 2"):
                    model_pertb_1 = get_params(model, model_pertb_1, top_eigenvector[0], l1)
                    model_pertb_2 = get_params(model_pertb_1, model_pertb_2, top_eigenvector[1], l2)
                    # loss = []
                    loss = 0.
                    for inputs, targets in dataloader_val:
                        logits = model_pertb_2(inputs)
                        outputs = logits.reshape(-1, logits.size(-1))
                        # loss.append(torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1))
                        loss += criterion(outputs, targets.reshape(-1)).item()
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
        plt.savefig(out_dir / (str(seed) + '.png'))
        seed = seed + 1

