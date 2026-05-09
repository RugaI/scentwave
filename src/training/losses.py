# -*- coding: utf-8 -*-
"""
ScentWave loss functions.

Total loss = lambda_vad * L_vad + lambda_gen * L_gen + lambda_div * L_div

L_vad  — MSE between predicted emotion and ground-truth VAD
L_gen  — KL divergence between predicted note distribution and target
L_div  — Diversity regulariser: penalise collapsed / monotone formulas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def vad_loss(pred_emotion: torch.Tensor, target_vad: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_emotion, target_vad)


def generation_loss(pred_formula: torch.Tensor,
                    target_notes: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """KL(target || pred) — drives formula to match expected scent profile."""
    log_pred = torch.log(pred_formula + eps)
    kl = F.kl_div(log_pred, target_notes + eps, reduction="batchmean", log_target=False)
    return kl


def diversity_loss(pred_formula: torch.Tensor) -> torch.Tensor:
    """
    Penalise collapsed formulas (all weight on one note).
    Maximise entropy of the note distribution.
    """
    eps = 1e-8
    entropy = -(pred_formula * torch.log(pred_formula + eps)).sum(dim=-1)
    return -entropy.mean()   # negative = we want to maximise entropy


class ScentWaveLoss(nn.Module):
    def __init__(self,
                 lambda_vad: float = 1.0,
                 lambda_gen: float = 2.0,
                 lambda_div: float = 0.1):
        super().__init__()
        self.lambda_vad = lambda_vad
        self.lambda_gen = lambda_gen
        self.lambda_div = lambda_div

    def forward(self,
                pred_emotion:  torch.Tensor,
                pred_formula:  torch.Tensor,
                target_vad:    torch.Tensor,
                target_notes:  torch.Tensor) -> dict:
        l_vad  = vad_loss(pred_emotion, target_vad)
        l_gen  = generation_loss(pred_formula, target_notes)
        l_div  = diversity_loss(pred_formula)
        total  = (self.lambda_vad * l_vad
                + self.lambda_gen * l_gen
                + self.lambda_div * l_div)
        return {
            "total":       total,
            "vad":         l_vad,
            "generation":  l_gen,
            "diversity":   l_div,
        }
