# dataset.py
import os
import pickle
import torch
from torch.utils.data import Dataset
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from utils import convert_state_to_tensor, move_str_to_idx
from dataset_rubik_seq import generate_single_case
from tokenizer.tokenizer_rubik import RubikTokenizer


class RubikDataset(Dataset):
    def __init__(self, data_dir='data', history_len=8, max_files=None,
                 num_samples=0,
                 min_scramble=8,
                 max_scramble=25):
        super().__init__()
        self.history_len = history_len
        self.SOS_token = SOS_TOKEN
        self.EOS_token = EOS_TOKEN
        self.tokenizer = RubikTokenizer()
        self._cache = {}  # 缓存各个文件的数据

        if data_dir is not None:
            pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            pkl_files.sort()
            if max_files is not None:
                pkl_files = pkl_files[:max_files]
            self.sample_index = []  # 每个元素为 (file_path, raw_item_index, t)
            for pf in pkl_files:
                full_path = os.path.join(data_dir, pf)
                # 加载整个文件的原始 item 列表（lazy load 模式下，这里仅加载索引信息）
                with open(full_path, 'rb') as f:
                    file_samples = []
                    try:
                        while True:
                            data_list = pickle.load(f)
                            file_samples.extend(data_list)
                    except EOFError:
                        pass
                # 构建该文件内每个原始 item 的滑动窗口索引
                for raw_item_idx, item in enumerate(file_samples):
                    steps = item['steps']
                    if len(steps) < self.history_len + 1:
                        continue
                    # 对每个 item，从 t = history_len 到 len(steps)-1 都生成一个样本
                    for t in range(self.history_len, len(steps)):
                        self.sample_index.append((full_path, raw_item_idx, t))
        else:
            # data_dir 为 None 时，继续使用原有的内存生成方式
            self.samples = []
            self._generate_in_memory(num_samples, min_scramble, max_scramble)

    def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
        print(f"RubikSeqDataset: 正在内存中生成 {num_samples} 条数据...")
        self.samples = []  # 先清空，以防重复追加
        for _ in range(num_samples):
            # 生成单条数据
            single_item = generate_single_case(min_scramble, max_scramble)
            steps = single_item['steps']
            # 如果 steps 为空，就无法生成任何样本，直接跳过
            if len(steps) == 0:
                continue

            # 针对每个 t，从 0 到 len(steps)-1 都生成一个样本
            for t in range(len(steps)):
                seq_len = self.history_len + 1

                # 先用 PAD_TOKEN 填充整个 src_seq
                src_seq = torch.full((seq_len, 55),
                                     PAD_TOKEN,
                                     dtype=torch.long)

                # 确定可用的真实 steps 范围：从 max(0, t-history_len) 到 t（含 t）
                start_idx = max(0, t - self.history_len)
                used_steps = steps[start_idx: t + 1]

                # 用于将真实 steps 数据填到 src_seq 的右侧
                offset = seq_len - len(used_steps)

                for i, (s6x9_i, mv_i) in enumerate(used_steps):
                    mv_i_idx = self.tokenizer.encode_move(mv_i)
                    state_tensor = self.tokenizer.encode_state(s6x9_i)

                    src_seq[offset + i, :54] = state_tensor
                    src_seq[offset + i, 54] = mv_i_idx

                # 构造 tgt：以 SOS 为起始符，后面跟从当前时刻 t 开始直到解法结束的 move 序列
                tgt_list = [self.SOS_token]
                for idx in range(t+1, len(steps)):
                    _, mv = steps[idx]
                    move_idx = self.tokenizer.encode_move(mv)
                    tgt_list.append(move_idx)
                tgt_list.append(self.EOS_token)
                tgt_seq = torch.tensor(tgt_list, dtype=torch.long)

                self.samples.append((src_seq, tgt_seq))
        print(f"RubikSeqDataset: 内存生成完毕，共生成 {len(self.samples)} 条数据.")

    def _load_from_file(self, file_path):
        """
        修改后的数据加载函数，生成 seq2seq 模型的训练样本。
        假设每个 item 中的 steps 为 [(s6x9_0, mv_0), (s6x9_1, mv_1), ...]，
        其中 s6x9_i 为状态信息（可通过 convert_state_to_tensor 转换为 shape (54,) 的张量），
        mv_i 为 move 字符串（通过 move_str_to_idx 转换为 move 索引）。

        生成样本方式：对每个 item 的 steps，从 t = history_len 到 len(steps)-1，
        生成一个样本，其中：
          - src: 从 steps[t-history_len] 到 steps[t]，每个时间步由 55 维向量构成（前 54 维为状态，最后1维为对应 move 索引，如果没有则为 -1）
          - tgt: 目标 move 序列，形如 [SOS, move[t], move[t+1], ..., move[len(steps)-1]]
                其中 SOS 为起始符（需要在类中定义 self.SOS_token）
        """
        import pickle
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    break
                for item in data_list:
                    steps = item['steps']
                    # 如果 steps 为空，就无法生成任何样本，直接跳过
                    if len(steps) == 0:
                        continue

                    # 针对每个 t，从 0 到 len(steps)-1 都生成一个样本
                    for t in range(len(steps)):
                        seq_len = self.history_len + 1

                        # 先用 PAD_TOKEN 填充整个 src_seq
                        src_seq = torch.full((seq_len, 55),
                                             PAD_TOKEN,
                                             dtype=torch.long)

                        # 确定可用的真实 steps 范围：从 max(0, t-history_len) 到 t（含 t）
                        start_idx = max(0, t - self.history_len)
                        used_steps = steps[start_idx: t + 1]

                        # 用于将真实 steps 数据填到 src_seq 的右侧
                        offset = seq_len - len(used_steps)

                        for i, (s6x9_i, mv_i) in enumerate(used_steps):
                            mv_i_idx = self.tokenizer.encode_move(mv_i)
                            state_tensor = self.tokenizer.encode_state(s6x9_i)

                            src_seq[offset + i, :54] = state_tensor
                            src_seq[offset + i, 54] = mv_i_idx

                        # 构造 tgt：以 SOS 为起始符，后面跟从当前时刻 t 开始直到解法结束的 move 序列
                        tgt_list = [self.SOS_token]
                        for idx in range(t+1, len(steps)):
                            _, mv = steps[idx]
                            move_idx = self.tokenizer.encode_move(mv)
                            tgt_list.append(move_idx)
                        tgt_list.append(self.EOS_token)
                        tgt_seq = torch.tensor(tgt_list, dtype=torch.long)

                        self.samples.append((src_seq, tgt_seq))

    def __len__(self):
        if hasattr(self, 'sample_index'):
            return len(self.sample_index)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, 'sample_index'):
            file_path, raw_item_idx, t = self.sample_index[idx]
            return self._load_sample(file_path, raw_item_idx, t)
        else:
            return self.samples[idx]

    def _load_sample(self, file_path, raw_item_idx, t):
        # 利用缓存避免重复加载同一文件的数据
        if file_path not in self._cache:
            with open(file_path, 'rb') as f:
                file_samples = []
                try:
                    while True:
                        data_list = pickle.load(f)
                        file_samples.extend(data_list)
                except EOFError:
                    pass
            self._cache[file_path] = file_samples
        else:
            file_samples = self._cache[file_path]
        item = file_samples[raw_item_idx]
        steps = item['steps']

        # 构造 src_seq：尺寸为 (history_len+1, 55)
        seq_len = self.history_len + 1
        src_seq = torch.empty((seq_len, 55), dtype=torch.long)
        for i, idx in enumerate(range(t - self.history_len, t + 1)):
            s6x9_i, mv_i = steps[idx]
            mv_i_idx = self.tokenizer.encode_move(mv_i)
            state_tensor = self.tokenizer.encode_state(s6x9_i)
            src_seq[i, :54] = state_tensor
            src_seq[i, 54] = mv_i_idx

        # 构造 tgt_seq：以 SOS 开始，后续为 t 到结束的 move 序列，最后追加 EOS
        tgt_list = [self.SOS_token]
        for idx in range(t+1, len(steps)):
            _, mv = steps[idx]
            move_idx = self.tokenizer.encode_move(mv)
            tgt_list.append(move_idx)
        tgt_list.append(self.EOS_token)
        tgt_seq = torch.tensor(tgt_list, dtype=torch.long)

        return (src_seq, tgt_seq)


from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    batch: list of (src_seq, tgt_seq)
      - src_seq.shape = (history_len+1, 55)
      - tgt_seq: 1D tensor of token indices, 长度可能不同
    """
    src_seqs = [x[0] for x in batch]
    tgt_seqs = [x[1] for x in batch]

    # 固定长度的 src 直接 stack
    src_tensor = torch.stack(src_seqs, dim=0)  # (B, history_len+1, 55)

    # 对 tgt_seqs 进行 padding，使它们具有相同长度
    tgt_tensor = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_TOKEN)

    return src_tensor, tgt_tensor
