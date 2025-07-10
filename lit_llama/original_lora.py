# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing_extensions import Self

from lit_llama.model import LLaMAConfig as BaseConfig
from lit_llama.model import LLaMA as BaseModel
from lit_llama.model import Block as BaseBlock
from lit_llama.model import CausalSelfAttention as BaseCausalSelfAttention
from lit_llama.model import MLP as BaseMLP
from lit_llama.model import RMSNorm
from lit_llama.model import MaskCache, RoPECache, KVCache, build_rope_cache
from lit_llama.utils import find_multiple
from lit_gpt.utils import map_old_state_dict_weights

from lit_gpt.original_lora import LoRALayer, LoRALinear, LoRAQKVLinear


@dataclass
class Config(BaseConfig):
    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    # all matrix implemented with LoRA
    to_query: bool = True
    to_key: bool = True
    to_value: bool = True
    to_projection: bool = True
    to_mlp: bool = True
    to_head: bool = True
    # lsq parameters


class LLaMA(BaseModel):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = LoRALinear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            r=(config.r if config.to_head else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            # lsq parameters
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []
        self.max_seq_length = self.config.block_size

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def forward(
        self, idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(Config.from_name(name))

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.config = config


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LoRAQKVLinear(
            in_features=config.n_embd,
            out_features=3*config.n_embd,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            enable_lora=(config.to_query, config.to_key, config.to_value),
            bias=config.bias,
            # for MQA/GQA support
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.c_proj = LoRALinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_projection else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None
        self.config = config
        self.landscape = False

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "c_attn.weight": "c_attn.linear.weight",
            "c_attn.bias": "c_attn.linear.bias",
            "c_proj.weight": "c_proj.linear.weight",
            "c_proj.bias": "c_proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class MLP(BaseMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.c_fc1 = LoRALinear(
            config.n_embd,
            n_hidden,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.c_fc2 = LoRALinear(
            config.n_embd,
            n_hidden,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.c_proj = LoRALinear(
            n_hidden,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "c_fc1.weight": "c_fc1.linear.weight",
            "c_fc1.bias": "c_fc1.linear.bias",
            "c_fc2.weight": "c_fc2.linear.weight",
            "c_fc2.bias": "c_fc2.linear.bias",
            "c_proj.weight": "c_proj.linear.weight",
            "c_proj.bias": "c_proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)



def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def enable_lsq_lora(model: LLaMA) -> None:
    """Transform LoRA modules to enable lsq"""
    lsq = True
    for name, module in model.named_modules():
        if not model.lm_head_quant and "lm_head" in name and isinstance(module, LoRALinear):
            pass
        elif isinstance(module, LoRALinear):
            module.lsq = lsq
            if isinstance(module, LoRAQKVLinear):
                module.lora_lsq_s.data = torch.max(torch.abs((module.linear.weight + module.conv1d(module.lora_A.data.unsqueeze(0), module.lora_B.data.unsqueeze(-1)).squeeze(0) * module.l_scale).detach().contiguous().view(3, -1, module.gs)), dim=-1)[0]/module.div
            else:
                module.lora_lsq_s.data = torch.max(torch.abs((module.linear.weight + (module.lora_B @ module.lora_A) * module.l_scale).detach().contiguous().view(-1, module.gs)), dim=-1)[0]/module.div
            module.lora_lsq_s.requires_grad = True
            module.lora_lsq_b.requires_grad = True


def merge_lsq_lora_weights(model: LLaMA) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    count = 0
    for name, module in model.named_modules():
        if not model.lm_head_quant and "lm_head" in name and isinstance(module, LoRALinear):
            print("lm_head with only lora params")
            module.w_bits = 0
            module.merge()
        elif isinstance(module, LoRALinear):
            count += module.linear.weight.numel()
            module.merge()
    print("reduced model size : ", count)


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


