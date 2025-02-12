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

    def __init__(self, data_dir='data', history_len=3, max_files=None):
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
                    # item['steps'] = [(s6x9_0, mv_0), (s6x9_1, mv_1), ...]
                    steps = item['steps']
                    # 我们只遍历到 len(steps)-1, 因为 steps[-1] 的动作可能是最后一步
                    for t in range(self.history_len, len(steps)):
                        # label = steps[t] 的 move
                        s6x9_label, mv_label = steps[t]
                        if mv_label is None:
                            # 如果最后一个是 None, 一般就跳过; 也可按需处理
                            continue

                        move_idx = move_str_to_idx(mv_label)

                        # 收集 [t-history_len, ..., t-1, t] 的 state/move
                        # e.g. t=3, history_len=3 => 0,1,2(包含move), 3(只包含state)
                        history_pairs = []
                        for i in range(t - self.history_len, t):
                            s6x9_i, mv_i = steps[i]
                            if mv_i is None:
                                # 可能是 steps[0] => (..., None)
                                # 用一个特定 label 或直接跳过
                                mv_i_idx = -1  # or any special token
                            else:
                                mv_i_idx = move_str_to_idx(mv_i)

                            # state shape= (54,)
                            state_tensor = convert_state_to_tensor(s6x9_i)
                            # 这里把 state 和 move 拼接
                            # e.g. shape= (54 + 1= 55)
                            pair_vec = torch.cat([
                                state_tensor,
                                torch.tensor([mv_i_idx], dtype=torch.long)
                            ], dim=0)  # => shape(55,)
                            history_pairs.append(pair_vec)

                        # 对于当前这一步 t 的 state, 不带 move
                        s_t_tensor = convert_state_to_tensor(s6x9_label)  # shape(54,)

                        # 现在 history_pairs 有 exactly history_len 个 (55,) 向量
                        # 我们也要加一个 “当前state(54)” => 也可以扩成 55带个dummy
                        # 这里演示就保持 54
                        # => total seq_len = history_len + 1

                        # 最后将它们打包 => shape (history_len+1, 55)?
                        # 需要对齐一下, 最后一个可以拼 dummy move = -1
                        # 这里举例: 就再 cat
                        pair_curr = torch.cat([
                            s_t_tensor,
                            torch.tensor([-1], dtype=torch.long)  # dummy move
                        ], dim=0)  # shape(55,)

                        full_seq = torch.stack(history_pairs + [pair_curr], dim=0)
                        # full_seq shape = (history_len+1, 55)

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

    # 这里简单 pad/cat, 如果history_len固定，就不用pad
    # seqs => (B, history_len+1, 55)
    seqs_tensor = torch.stack(seqs, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return seqs_tensor, labels_tensor
