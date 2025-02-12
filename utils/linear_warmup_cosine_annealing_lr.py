# utils/linear_warmup_cosine_annealing_lr.py
import math
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        """
        参数说明：
          optimizer         优化器实例
          warmup_epochs     预热（warmup）阶段的epoch数
          max_epochs        总共的epoch数（必须大于或等于warmup_epochs，否则后续cosine annealing阶段无法正确计算）
          warmup_start_lr   预热初始学习率
          eta_min           学习率衰减下限
          last_epoch        上一个epoch索引，默认-1
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 在warmup阶段，学习率线性增长
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        # 在warmup之后，采用cosine annealing衰减
        else:
            t = self.last_epoch - self.warmup_epochs
            T = self.max_epochs - self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T)) / 2
                for base_lr in self.base_lrs
            ]
