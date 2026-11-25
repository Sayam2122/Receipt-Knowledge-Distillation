"""Compact student: shared MLP trunk + one scoring head per field (~18k parameters)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sroie_data import FIELDS


class CandidateStudent(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(h2, 1) for _ in FIELDS])

    def forward(self, x, field_id):                      # x [B,K,D], field_id [B] -> scores [B,K]
        h = self.trunk(x)
        out = torch.stack([hd(h).squeeze(-1) for hd in self.heads], 1)   # [B,F,K]
        return out[torch.arange(x.shape[0], device=x.device), field_id]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def kd_loss(scores, mask, tmask, tprob, lam, T=1.0, weight=None):
    """Per-group loss.
    scores [B,K]; mask [B,K] valid candidate; tmask [B,K] ground-truth candidates (all False = annotation
    not available for this group); tprob [B,K] teacher distribution (all zeros = no teacher); lam in [0,1].
      annotated + teacher : (1-lam)*CE + lam*T^2*KL(q || p_T)
      annotated only      : CE
      teacher only        : T^2*KL(q || p_T)        (label-free distillation)
    weight [B] optional per-group KL weight. Groups with neither signal are ignored."""
    neg = -1e4
    s = scores.masked_fill(~mask, neg)
    logp = F.log_softmax(s, -1)
    has_t = tmask.any(-1).float()
    ce = -torch.logsumexp(logp.masked_fill(~tmask, neg), -1) * has_t
    has_q = (tprob.sum(-1) > 0).float()
    q = tprob / tprob.sum(-1, keepdim=True).clamp_min(1e-12)
    logpT = F.log_softmax(s / T, -1)
    kl = (torch.special.xlogy(q, q) - q * logpT).sum(-1) * T * T
    w = has_q if weight is None else has_q * weight
    ce_w = has_t * (1 - lam * has_q)
    kl_w = (lam * has_t + (1 - has_t)) * w
    n = ((has_t + has_q) > 0).float().sum()
    return (ce_w * ce + kl_w * kl).sum() / n.clamp_min(1.0)
