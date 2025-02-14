# dataset.py
import os
import pickle
import torch
from torch.utils.data import Dataset

from utils import convert_state_to_tensor, move_str_to_idx


class RubikDataset(Dataset):
    """
    带历史的Dataset: 返回:
      [ (state_{t-history_len}, move_{t-history_len}),
        ...
        (state_{t-1}, move_{t-1}),
        state_t
      ],  label = move_t
    """

    def __init__(self, data_dir='data', history_len=8, max_files=None):
        super().__init__()
        self.samples = []
        self.history_len = history_len

        pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        pkl_files.sort()
        if max_files is not None:
            pkl_files = pkl_files[:max_files]

        for pf in pkl_files:
            full_path = os.path.join(data_dir, pf)
            self._load_from_file(full_path)

    def _load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    break
                for item in data_list:
                    # 每个 item 中，steps 为 [(s6x9_0, mv_0), (s6x9_1, mv_1), ...]
                    steps = item['steps']
                    # 遍历时从 history_len 开始，保证有足够的历史记录
                    for t in range(self.history_len, len(steps)):
                        s6x9_label, mv_label = steps[t]
                        # 当前 label 的 move 索引
                        move_idx = move_str_to_idx(mv_label)
                        seq_len = self.history_len + 1
                        # 预分配张量，shape = (history_len+1, 55)，dtype 使用 long 类型（假设状态已转换为离散表示）
                        full_seq = torch.empty((seq_len, 55), dtype=torch.long)

                        # 填充历史记录部分：从 t-history_len 到 t-1 的每一步
                        for i, idx in enumerate(range(t - self.history_len, t)):
                            s6x9_i, mv_i = steps[idx]
                            # 如果 mv_i 为 None，使用 -1 表示特殊 token
                            mv_i_idx = move_str_to_idx(mv_i) if mv_i is not None else -1
                            state_tensor = convert_state_to_tensor(s6x9_i)  # shape (54,)
                            # 将 state_tensor 填入前 54 个元素
                            full_seq[i, :54] = state_tensor
                            # 最后一个位置填入当前步对应的 move 索引（或 -1）
                            full_seq[i, 54] = mv_i_idx

                        # 最后一行为当前状态，不附带 move 信息，填入 dummy 值 -1
                        s_t_tensor = convert_state_to_tensor(s6x9_label)  # shape (54,)
                        full_seq[self.history_len, :54] = s_t_tensor
                        full_seq[self.history_len, 54] = -1  # dummy move

                        self.samples.append((full_seq, move_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    batch: list of (full_seq, label)
      full_seq.shape = (history_len+1, 55)
      label => int
    """
    seqs = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    # 直接 stack，因为每个样本的尺寸固定
    seqs_tensor = torch.stack(seqs, dim=0)  # shape (B, history_len+1, 55)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return seqs_tensor, labels_tensor
