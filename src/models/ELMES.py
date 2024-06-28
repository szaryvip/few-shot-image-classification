"""ELMES implementation by authors of the CAML paper:
Christopher Fifty, Dennis Duan, Ronald G. Junkins,
Ehsan Amid, Jure Leskovec, Christopher Ré, Sebastian Thrun
https://arxiv.org/abs/2310.10971
"""
import numpy as np
import torch


def get_elmes(p, C):
    ones = torch.ones((C, 1), dtype=torch.float32)
    M_star = torch.sqrt(torch.tensor(C / (C - 1))) * (
            torch.eye(C) - 1 / C * torch.matmul(ones, torch.transpose(ones, 0, 1)))
    np.random.seed(50)
    U = np.random.random(size=(p, C))
    U = torch.tensor(np.linalg.qr(U)[0][:, :C]).to(torch.float32)
    return (U @ M_star).T
