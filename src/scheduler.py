import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineDecayScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.0, last_epoch=-1):
        """
        Implements a learning rate scheduler with linear warmup followed by cosine decay.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of warmup epochs.
            total_epochs (int): Total number of epochs (including warmup).
            eta_min (float): Minimum learning rate after cosine decay. Default: 0.0.
            last_epoch (int): The index of the last epoch. Default: -1 (start fresh).
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]

# CAML: 9600 warmups, 360.000 decay (steps) - 1:37.5 ratio
# PMF: 5 warmups, 95 decay (epochs) - 1:19 ratio
# min lr 1e-6 max lr 1e-5
