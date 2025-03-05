# dataset_rubik_pact.py
import os
import pickle
import torch
from torch.utils.data import Dataset

class RubikDatasetPACT(Dataset):
    """
    和原来的 RubikDataset 类似，但去掉了对 RubikTokenizer 的依赖。
    不再在这里进行 state/move 的编码，而是直接返回 (src_raw, tgt_raw) 的原始信息。
    """

    def __init__(self, data_dir='data', history_len=8, max_files=None,
                 num_samples=0,
                 min_scramble=8,
                 max_scramble=25):
        super().__init__()
        self.samples = []
        self.history_len = history_len

        if data_dir is not None:
            pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            pkl_files.sort()
            if max_files is not None:
                pkl_files = pkl_files[:max_files]
            for pf in pkl_files:
                full_path = os.path.join(data_dir, pf)
                self._load_from_file(full_path)
        else:
            # 如果没有 data_dir，就走内存生成逻辑
            self._generate_in_memory(num_samples, min_scramble, max_scramble)

    def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
        """
        如果你有 generate_single_case，可以在这里得到 steps = [(s6x9, move), ...]。
        只做滑窗切分，记录原始数据，而不做 encode。
        """
        from dataset_rubik_seq import generate_single_case  # 如果有这个函数
        print(f"RubikDatasetPACT: 正在内存中生成 {num_samples} 条数据...")
        for _ in range(num_samples):
            item = generate_single_case(min_scramble, max_scramble)
            steps = item['steps']  # [(state, move), ...]
            if len(steps) < self.history_len + 1:
                continue
            for t in range(self.history_len, len(steps)):
                # 构造原始 src
                src_seq = steps[t - self.history_len : t + 1]  # [(s6x9_i, mv_i), ...]

                # 构造原始 tgt
                # 注意，这里不再插入 SOS/ EOS，也不 encode move
                tgt_list = [mv for (_, mv) in steps[t:]]  # 当前时刻到结尾
                # 存储 (src_seq, tgt_list)
                self.samples.append((src_seq, tgt_list))

        print(f"RubikDatasetPACT: 内存生成完毕，共生成 {len(self.samples)} 条数据.")

    def _load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    break
                for item in data_list:
                    steps = item['steps']  # [(s6x9_i, mv_i), ...]
                    if len(steps) < self.history_len + 1:
                        continue
                    for t in range(self.history_len, len(steps)):
                        src_seq = steps[t - self.history_len : t + 1]
                        tgt_list = [mv for (_, mv) in steps[t:]]
                        self.samples.append((src_seq, tgt_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回的 src_seq 是一个 list，里面每个元素=(s6x9_str, move_str or None)
        返回的 tgt_seq 是一个 list，里面是 move_str or None
        """
        return self.samples[idx]
