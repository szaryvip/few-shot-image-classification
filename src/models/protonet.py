"""Implementation by authors of the P>M>F paper:
Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, Timothy M. Hospedales
https://arxiv.org/pdf/2204.07305
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import DiffAugment


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        self.backbone = backbone

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2)
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1

        B, nSupp, C, H, W = supp_x.shape
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)

        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)

        feat = self.backbone.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1)

        logits = self.cos_classifier(prototypes, feat)
        return logits


class ProtoNet_Finetune(ProtoNet):
    def __init__(self, backbone, num_iters=50, lr=5e-2, aug_prob=0.9,
                 aug_types=['color', 'translation']):
        super().__init__(backbone)
        self.num_iters = num_iters
        self.lr = lr
        self.aug_types = aug_types
        self.aug_prob = aug_prob

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        state_dict = self.backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        # reset backbone state
        self.backbone.load_state_dict(self.backbone_state, strict=True)

        if self.lr == 0:
            return super().forward(supp_x, supp_y, x)

        B, nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1
        device = x.device

        criterion = nn.CrossEntropyLoss()
        supp_x = supp_x.view(-1, C, H, W)
        x = x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)
        supp_y = supp_y.view(-1)

        opt = torch.optim.Adam(self.backbone.parameters(),
                               lr=self.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)

        def single_step(z, mode=True):
            '''
            z = Aug(supp_x) or x
            '''
            with torch.set_grad_enabled(mode):
                supp_f = self.backbone.forward(supp_x)
                supp_f = supp_f.view(B, nSupp, -1)
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)

                feat = self.backbone.forward(z)
                feat = feat.view(B, z.shape[0], -1)

                logits = self.cos_classifier(prototypes, feat)
                loss = None

                if mode:
                    loss = criterion(logits.view(B * nSupp, -1), supp_y)

            return logits, loss

        for i in range(self.num_iters):
            opt.zero_grad()
            z = DiffAugment(supp_x, self.aug_types, self.aug_prob, detach=True)
            _, loss = single_step(z, True)
            loss.backward()
            opt.step()

        logits, _ = single_step(x, False)
        return logits
