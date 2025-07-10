import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class L4Quantizer(Function):
    @staticmethod
    def forward(ctx, x, W, bias, A, B, l_scale, step, qbias, gs, Qn, Qp, g):
        '''
        For this work, each layer of weights and each layer of activations has a distinct step size, represented
        as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
        batch of activations, respectively
        '''
        # x, W, A, B, step, qbias = x.to(torch.bfloat16), W.to(torch.bfloat16), A.to(torch.bfloat16), B.to(torch.bfloat16), step.to(torch.bfloat16), qbias.to(torch.bfloat16)
        ctx.save_for_backward(x, W, A, B, step, qbias)
        ctx.other = l_scale, gs, Qn, Qp, g
        # Quantize by scale
        weight = (W + B @ A * l_scale)  # (o, i)
        step = step.unsqueeze(0).repeat(gs, 1).T.contiguous().view(W.size())  # (gs, o*gn) -> (o*gn, gs) -> (o, i)
        qbias = qbias.unsqueeze(0).repeat(gs, 1).T.contiguous().view(W.size())
        w_q = torch.div(weight - qbias, step).clamp(Qn, Qp).round() * step + qbias
        del step, qbias

        mm = x @ w_q.T
        if bias is not None:
            return mm + torch.broadcast_to(bias, mm.size())
        else:
            return mm

    @staticmethod
    def backward(ctx, grad_out):
        x, W, A, B, step, qbias = ctx.saved_tensors
        l_scale, gs, Qn, Qp, g = ctx.other
        weight = (W + B @ A * l_scale)  # (o, i)
        step = step.unsqueeze(0).repeat(gs, 1).T.contiguous().view(W.size())  # (gs, o*gn) -> (o*gn, gs) -> (o, i)
        qbias = qbias.unsqueeze(0).repeat(gs, 1).T.contiguous().view(W.size())
        q_w = torch.div(weight - qbias, step)

        between = q_w.ge(Qn) & q_w.le(Qp)
        q_w = q_w.clamp(Qn, Qp)

        grad_in = grad_out @ (q_w.round() * step + qbias)  # (1, s, o) @ (o, i) -> (1, s, i)
        del step, qbias
        grad_w = grad_out[0].T @ x[0]  # (o, s) @ (s, i) -> (o, i)
        grad_wT = torch.where(between, grad_w, 0.0).T  # (i, o)
        grad_a = (grad_wT @ B * l_scale).T  # (i, o) @ (o, r) -> (i, r) -> (r, i)
        grad_b = (A @ grad_wT * l_scale).T  # (r, i) @ (i, o) -> (r, o) -> (o, r)

        grad_s = (torch.where(between, (q_w.round() - q_w), q_w) * grad_w).contiguous().view(-1, gs).T.sum(dim=0) * g # (o, i) -> (o*gn, gs) -> (gs, o*gn)
        grad_qb = (torch.where(between, 0.0, grad_w)).T.contiguous().view(-1, gs).T.sum(dim=0) * g

        return grad_in, None, None, grad_a, grad_b, None, grad_s, grad_qb, None, None, None, None


