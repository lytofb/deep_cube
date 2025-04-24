import math
import torch
from torch.utils.data import Sampler
from torch.distributed import get_world_size, get_rank

class DistributedWeightedSampler(Sampler):
    """
    与官方 DistributedSampler 类似，但在每个 epoch 内按照给定权重做随机采样。
    """
    def __init__(self, weights, num_samples=None, replacement=True, drop_last=False):
        """
        Args:
            weights (Tensor): shape (N,) 的采样权重（倒频率）
            num_samples (int): 每个进程要采多少个样本。如果 None，就平均分配。
            replacement (bool): torch.multinomial 的 replacement
        """
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples_total = len(self.weights)
        self.num_replicas = get_world_size()
        self.rank = get_rank()
        self.replacement = replacement
        self.drop_last = drop_last

        if num_samples is None:
            # 保证所有 rank 总和 ≈ dataset 大小
            self.num_samples = int(math.ceil(self.num_samples_total / self.num_replicas))
        else:
            self.num_samples = num_samples

        self.epoch = 0

    def __iter__(self):
        # 全局同一个随机种子 + epoch，让每轮重新洗牌
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # torch.multinomial 会按权重采 self.num_samples_total 个索引（可重复）
        indices = torch.multinomial(
            self.weights,
            self.num_samples_total if self.replacement else self.num_samples_total,
            self.replacement,
            generator=g
        ).tolist()

        # 再按 rank 拆分
        indices = indices[self.rank : len(indices) : self.num_replicas]

        # 如果不足 num_samples，用补齐或截断
        if len(indices) < self.num_samples:
            indices += indices[: (self.num_samples - len(indices))]
        else:
            indices = indices[: self.num_samples]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
