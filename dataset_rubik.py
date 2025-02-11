# dataset.py
import os
import pickle
import torch
from torch.utils.data import Dataset

from utils import convert_state_to_tensor, move_str_to_idx


class RubikDataset(Dataset):
    """
    从 data/ 下多个 pkl 文件中加载 (state_6x9, move) 对，
    转成 (input_tensor, label) 用于监督学习。
    """

    def __init__(self, data_dir='rubik_shards', max_files=None):
        """
        data_dir: 存放 part_****.pkl 的文件夹
        max_files: 只加载前 max_files 个 pkl, 以免一次性加载太大
        """
        self.samples = []  # 存 (state_tensor, move_idx)
        pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        pkl_files.sort()
        if max_files is not None:
            pkl_files = pkl_files[:max_files]

        for pf in pkl_files:
            full_path = os.path.join(data_dir, pf)
            self._load_from_file(full_path)

    def _load_from_file(self, file_path):
        """
        注意：如果 pkl 文件是多段写入，需要多次 load 直到 EOF。
        否则若是一次性保存的列表，可直接 load 一次。
        """
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_list = pickle.load(f)  # 这是一个列表 of items
                except EOFError:
                    break
                # data_list 是一批 item, item 结构参考生成脚本
                for item in data_list:
                    # item = {'scramble','solution','steps'}
                    steps = item['steps']  # 例如 [(s6x9_0, None), (s6x9_1, M1), ...]

                    # 我们只遍历到 len(steps)-2，因为 i+1 不能越界
                    # 具体要不要用到最后一个状态，取决于你的需求
                    for i in range(len(steps) - 1):
                        s6x9_this, mv_this = steps[i]
                        s6x9_next, mv_next = steps[i + 1]

                        # 当前状态
                        state_tensor = convert_state_to_tensor(s6x9_this)

                        # 以下一步的 move 作为标签
                        # 如果 mv_next is None，代表下一步没有动作(比如已经最后一步了？) => 可以选择 continue 或者特判
                        if mv_next is None:
                            continue

                        move_idx = move_str_to_idx(mv_next)

                        # 将 (当前状态, 下一步动作) 加入训练样本
                        self.samples.append((state_tensor, move_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state_tensor, move_idx = self.samples[idx]
        return state_tensor, move_idx


def collate_fn(batch):
    """
    用于 DataLoader 的批处理函数。
    batch: list of (state_tensor, move_idx)
    """
    # state_tensor.shape = [54], move_idx是int
    # 如果用embedding，就需要 shape=(batch_size, 54).
    states = torch.stack([x[0] for x in batch], dim=0)  # (B, 54)
    moves = torch.tensor([x[1] for x in batch], dtype=torch.long)  # (B,)

    return states, moves
