import torch
import torch.nn as nn

from transformers import LlamaForCausalLM
from peft import PeftModel, set_peft_model_state_dict
from peft.tuners.lora import QuantLinear

import argparse
from typing import Union, Optional

def dequant(self):
    if self.bits in [2,4,8]:
        zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits), self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(zeros, (2 ** self.bits) - 1)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight,(2 ** self.bits) - 1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
        zeros = (zeros >> self.wf.unsqueeze(0))
        zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
        zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
        zeros = zeros & 0x7
        zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1))&0x7
        weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
        weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = (scales * (weight - zeros))
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight

def dt_analysis(model: Union[LlamaForCausalLM, PeftModel], verbose: bool):
    delta = {}
    dt_s_value, dt_m_value = 0.0, 0.0
    dt_s_num, dt_m_num = 0, 0
    for name, module in model.named_modules():
        if name == "lm_head":
            pass
        elif isinstance(module, QuantLinear):
            m = name.split('.')
            layer = m[2]
            target = m[-2]+'.'+m[-1]

            if layer not in delta:
                # print("Layer: ", layer)
                delta[layer] = {}

            weight = dequant(module.quant_linear_module)
            weight_m = weight.mean()

            lora_dt = module.lora_A.default.weight.transpose(0, 1) @ module.lora_B.default.weight.transpose(0, 1) * module.lora_alpha['default']
            lora_dt_absm = lora_dt.abs().mean()

            weight_m_dt = (lora_dt.mean() / weight.abs().mean() * 100).abs()
            delta[layer][target] = {"weight_m_old" : weight_m.data, "weight_m" : weight_m_dt.data, "weight" : lora_dt_absm.data}
            dt_m_value += weight_m_dt.data
            dt_m_num += 1

    print(f"mean weight delta : {dt_m_value / dt_m_num:.2e}")

    if verbose:
        targets = list(delta['0'].keys())
        print(targets)
        torch.set_printoptions(precision=2)
        torch.set_printoptions(sci_mode=True)
        delta_w = {}
        for target in targets:
            delta_w[target] = sorted(delta.items(), key=lambda x:x[1][target]['weight_m'].mean())
        print("****** Weight delta ******")
        for i in range(len(delta.keys())):
            print(f"layer {delta_w[targets[0]][i][0]:<2} {delta_w[targets[0]][i][1][targets[0]]['weight_m']:.2e} | ",
                  f"layer {delta_w[targets[1]][i][0]:<2} {delta_w[targets[1]][i][1][targets[1]]['weight_m']:.2e} | ",
                  f"layer {delta_w[targets[2]][i][0]:<2} {delta_w[targets[2]][i][1][targets[2]]['weight_m']:.2e} | ",
                  f"layer {delta_w[targets[3]][i][0]:<2} {delta_w[targets[3]][i][1][targets[3]]['weight_m']:.2e} | ",
                  f"layer {delta_w[targets[4]][i][0]:<2} {delta_w[targets[4]][i][1][targets[4]]['weight_m']:.2e}"
                  )
        print("****** LoRA delta ******")
        for i in range(len(delta.keys())):
            print(f"layer {delta_w[targets[0]][i][0]:<2} {delta_w[targets[0]][i][1][targets[0]]['weight']:.2e} | ",
                  f"layer {delta_w[targets[1]][i][0]:<2} {delta_w[targets[1]][i][1][targets[1]]['weight']:.2e} | ",
                  f"layer {delta_w[targets[2]][i][0]:<2} {delta_w[targets[2]][i][1][targets[2]]['weight']:.2e} | ",
                  f"layer {delta_w[targets[3]][i][0]:<2} {delta_w[targets[3]][i][1][targets[3]]['weight']:.2e} | ",
                  f"layer {delta_w[targets[4]][i][0]:<2} {delta_w[targets[4]][i][1][targets[4]]['weight']:.2e}"
                  )



    return delta



def analysis(args):
    model = LlamaForCausalLM.from_pretrained(args.model).to("cpu")
    if args.peft:
        model = PeftModel.from_pretrained(model, args.peft).model

    delta = dt_analysis(model, args.verbose)


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="huggingface model directory")
    parser.add_argument("--peft", type=str, help="adapter directory")
    parser.add_argument("--verbose", type=bool, default=False, help="layer wise delta analysis")

    return parser.parse_args()



if __name__ == "__main__":
    args = parsing_args()
    if args.peft:
        print(f"Model: {args.model} | Adapter: {args.peft}")
    else:
        print(f"Model: {args.model}")

    analysis(args)
