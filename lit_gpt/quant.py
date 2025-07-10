import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class LSQP_WQuantizer(Function):
    @staticmethod
    def forward(ctx, x, W, bias, A, B, l_scale, step, qbias,
                gs, g, Qn, Qp):
        '''
        For this work, each layer of weights and each layer of activations has a distinct step size, represented
        as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
        batch of activations, respectively
        '''
        ctx.save_for_backward(x, W, A, B, step, qbias)
        ctx.other = l_scale, gs, g, Qn, Qp
        # Quantize by scale
        weight = W.transpose(0, 1) + A.transpose(0, 1) @ B.transpose(0, 1) * l_scale  # (i, o)
        step = step.unsqueeze(0).repeat(gs, 1).transpose(0, 1).contiguous().view(W.size()).transpose(0, 1)  # (gs, o*gn) -> (o*gn, gs) -> (o, i) -> (i, o)
        qbias = qbias.unsqueeze(0).repeat(gs, 1).transpose(0, 1).contiguous().view(W.size()).transpose(0, 1)
        w_q = torch.div(weight - qbias, step).clamp(Qn, Qp).round() * step + qbias
        del step, qbias

        mm = x @ w_q
        if bias is not None:
            return mm + torch.broadcast_to(bias, mm.size())
        else:
            return mm

    @staticmethod
    def backward(ctx, grad_out):
        x, W, A, B, step, qbias = ctx.saved_tensors
        l_scale, gs, g, Qn, Qp = ctx.other
        weight = W.transpose(0, 1) + A.transpose(0, 1) @ B.transpose(0, 1) * l_scale  # (i, o)
        step = step.unsqueeze(0).repeat(gs, 1).transpose(0, 1).contiguous().view(W.size()).transpose(0, 1)  # (gs, o*gn) -> (o*gn, gs) -> (o, i) -> (i, o)
        qbias = qbias.unsqueeze(0).repeat(gs, 1).transpose(0, 1).contiguous().view(W.size()).transpose(0, 1)
        q_w = torch.div(weight - qbias, step)

        between = q_w.ge(Qn) & q_w.le(Qp)
        q_w = q_w.clamp(Qn, Qp)

        grad_in = grad_out @ (q_w.round() * step).transpose(0, 1)  # (i, o) -> (o, i)  # (1, s, o) @ (o, i) -> (1, s, i)
        del step, qbias
        grad_w = (x[0].transpose(0, 1)) @ grad_out[0]  # (s, i) -> (i, s) @ (s, o) -> (i, o)
        grad_wT = torch.where(between, grad_w, 0.0).transpose(0, 1)  # (o, i)
        grad_a = B.transpose(0, 1) @ grad_wT * l_scale  # (r, o) @ (o, i) -> (r, i)
        grad_b = grad_wT @ A.transpose(0, 1) * l_scale  # (o, i) @ (i, r) -> (o, r)
        # grad_a = l_scale * (B.transpose(0, 1) @ grad_out[0].transpose(0, 1)) @ x[0]  # (r, o) @ (o, i) -> (r, i)
        # grad_b = l_scale * grad_out[0].transpose(0, 1) @ (x[0] @ A.transpose(0, 1))  # (o, i) @ (i, r) -> (o, r)

        grad_s = (torch.where(between, (q_w.round() - q_w), q_w) * grad_w * g).transpose(0, 1).contiguous().view(-1, gs).transpose(0, 1).sum(dim=0)  # (i, o) -> (o, i) -> (o*gn, gs) -> (gs, o*gn)
        grad_qb = (torch.where(between, 0.0, grad_w) * g).transpose(0, 1).contiguous().view(-1, gs).transpose(0, 1).sum(dim=0)

        return grad_in, None, None, grad_a, grad_b, None, grad_s, grad_qb, None, None, None, None, None

class LSQP_WQuantizerQKV(Function):
    @staticmethod
    def forward(ctx, x, W, bias, A, B, l_scale, step, qbias,
                gs, g, Qn, Qp, enable_lora):
        '''
        For this work, each layer of weights and each layer of activations has a distinct step size, represented
        as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
        batch of activations, respectively
        '''
        ctx.save_for_backward(x, W, A, B, step, qbias)
        ctx.other = l_scale, gs, g, Qn, Qp, enable_lora
        # Padding should be added for none-all lora case: enable_lora
        w = W.view(3, -1, W.shape[-1])  # (3*o, i) -> (3, o, i)
        a = A.view(3, -1, A.shape[-1])  # (3*r, i) -> (3, r, i)
        b = B.view(3, -1, B.shape[-1])  # (3*o, r) -> (3, o, r)
        weight = w.transpose(1, 2) + a.transpose(1, 2) @ b.transpose(1, 2) * l_scale  # (3, i, o)
        step = step.unsqueeze(1).repeat(1, gs, 1).transpose(1, 2).contiguous().view(w.size()).transpose(1, 2)  # (3, o*gn) -> (3, 1, o*gn) -> (3, gs, o*gn) -> (3, o*gn, gs) -> (3, o, i) -> (3, i, o)
        qbias = qbias.unsqueeze(1).repeat(1, gs, 1).transpose(1, 2).contiguous().view(w.size()).transpose(1, 2)
        w_q = torch.div(weight - qbias, step).clamp(Qn, Qp).round() * step + qbias
        del step, qbias

        mm = torch.cat((x@w_q[0], x@w_q[1], x@w_q[2]), dim=2)
        if bias is not None:
            return mm + torch.broadcast_to(bias, mm.size())
        else:
            return mm

    @staticmethod
    def backward(ctx, grad_out):
        x, W, A, B, step, qbias = ctx.saved_tensors
        l_scale, gs, g, Qn, Qp, enable_lora = ctx.other
        # Padding should be added for none-all lora case: enable_lora
        w = W.view(3, -1, W.shape[-1])  # (3*o, i) -> (3, o, i)
        a = A.view(3, -1, A.shape[-1])  # (3*r, i) -> (3, r, i)
        b = B.view(3, -1, B.shape[-1])  # (3*o, r) -> (3, o, r)
        weight = w.transpose(1, 2) + a.transpose(1, 2) @ b.transpose(1, 2) * l_scale  # (3, i, o)
        del w
        step = step.unsqueeze(1).repeat(1, gs, 1).transpose(1, 2).contiguous().view(3, -1, a.shape[-1]).transpose(1, 2)  # (3, o*gn) -> (3, 1, o*gn) -> (3, gs, o*gn) -> (3, o*gn, gs) -> (3, o, i) -> (3, i, o)
        qbias = qbias.unsqueeze(1).repeat(1, gs, 1).transpose(1, 2).contiguous().view(3, -1, a.shape[-1]).transpose(1, 2)
        q_w = torch.div(weight - qbias, step)

        between = q_w.ge(Qn) & q_w.le(Qp)
        q_w = q_w.clamp(Qn, Qp)

        grad_in = grad_out @ (q_w.round() * step).transpose(1, 2).contiguous().view(-1, q_w.shape[1])  # (3, i, o) -> (3, o, i) -> (3*o, i)  # (1, s, 3*o) @ (3*o, i) -> (1, s, i)
        del step, qbias
        grad_w = (grad_out[0].transpose(0, 1) @ x[0]).contiguous().view(3, -1, q_w.shape[1]).transpose(1,2)  # (s, 3*o) -> (3*o, s) @ (s, i) -> (3*o, i) -> (3, o, i) -> (3, i, o)
        grad_wT = (between * grad_w).transpose(1, 2)  # (3, i, o) -> (3, o, i)
        grad_wT = grad_w.transpose(1, 2)  # (3, i, o) -> (3, o, i)
        grad_a = (b.transpose(1, 2) @ grad_wT * l_scale).contiguous().view(-1, a.shape[-1])  # (3, r, o) @ (3, o, i) -> (3, r, i) -> (3*r, i)
        grad_b = (grad_wT @ a.transpose(1, 2) * l_scale).contiguous().view(-1, b.shape[-1])  # (3, o, i) @ (3, i, r) -> (3, o, r) -> (3*o, r)

        grad_s = (torch.where(between, (q_w.round() - q_w), q_w) * grad_w * g).contiguous().transpose(1, 2).contiguous().view(3, -1, gs).transpose(1,2).sum(dim=1)  # (3, i, o) -> (3, o, i) -> (3, o*gn, gs) ->  (3, gs, o*gn) -> (3, o*gn)
        grad_qb = (torch.where(between, 0.0, grad_w) * g).contiguous().transpose(1, 2).contiguous().view(3, -1, gs).transpose(1,2).sum(dim=1)

        return grad_in, None, None, grad_a, grad_b, None, grad_s, grad_qb, None, None, None, None, None, None

