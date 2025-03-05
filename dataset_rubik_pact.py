# dataset.py
import os
import pickle
import torch
from torch.utils.data import Dataset
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from utils import convert_state_to_tensor, move_str_to_idx
from dataset_rubik_seq import generate_single_case
from tokenizer.tokenizer_rubik import RubikTokenizer

class RubikDatasetPACT(Dataset):
    """
    带历史的Dataset: 返回:
      [ (state_{t-history_len}, move_{t-history_len}),
        ...
        (state_{t-1}, move_{t-1}),
        state_t
      ],  label = move_t
    """

    def __init__(self, data_dir='data', history_len=8, max_files=None,
                 num_samples=0,
                 min_scramble=8,
                 max_scramble=25):
        super().__init__()
        self.samples = []
        self.history_len = history_len
        self.SOS_token = SOS_TOKEN
        self.EOS_token = EOS_TOKEN
        self.tokenizer = RubikTokenizer()

        if data_dir is not None:
            pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            pkl_files.sort()
            if max_files is not None:
                pkl_files = pkl_files[:max_files]
            for pf in pkl_files:
                full_path = os.path.join(data_dir, pf)
                self._load_from_file(full_path)
        else:
            self._generate_in_memory(num_samples, min_scramble, max_scramble)

    def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
        print(f"RubikSeqDataset: 正在内存中生成 {num_samples} 条数据...")
        for _ in range(num_samples):
            # 生成单条数据
            single_item = generate_single_case(min_scramble, max_scramble)
            steps = single_item['steps']
            # 若总步数不足 history_len+1，则跳过
            if len(steps) < self.history_len + 1:
                continue
            # 对每个解法序列，采用滑动窗口生成多个样本
            for t in range(self.history_len, len(steps)):
                # 构造 src：选取从 t-history_len 到 t（包含 t）的连续状态
                seq_len = self.history_len + 1
                src_seq = torch.empty((seq_len, 55), dtype=torch.long)
                for i, idx in enumerate(range(t - self.history_len, t + 1)):
                    s6x9_i, mv_i = steps[idx]
                    # 如果 mv_i 为 None，用 -1 表示特殊 token
                    mv_i_idx = self.tokenizer.encode_move(mv_i)
                    state_tensor = self.tokenizer.encode_state(s6x9_i)
                    # 前54个位置为状态表示
                    src_seq[i, :54] = state_tensor
                    # 第55个位置记录该步的 move 索引（如果有的话）
                    src_seq[i, 54] = mv_i_idx

                # 假设上面已经生成了 src_seq，形状 (seq_len, 55)，其中 seq_len = self.history_len + 1
                tgt_seq = torch.empty(seq_len, dtype=torch.long)
                # 对于前 seq_len-1 个位置，用 "下一行的动作" 填充
                for i in range(seq_len - 1):
                    # src_seq[i+1, 54] 就是下一时刻的动作
                    tgt_seq[i] = src_seq[i + 1, 54]
                # 最后一项用 EOS
                tgt_seq[seq_len - 1] = self.EOS_token

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
                    # 若总步数不足 history_len+1，则跳过
                    if len(steps) < self.history_len + 1:
                        continue
                    # 对每个解法序列，采用滑动窗口生成多个样本
                    for t in range(self.history_len, len(steps)):
                        # 构造 src：选取从 t-history_len 到 t（包含 t）的连续状态
                        seq_len = self.history_len + 1
                        src_seq = torch.empty((seq_len, 55), dtype=torch.long)
                        for i, idx in enumerate(range(t - self.history_len, t + 1)):
                            s6x9_i, mv_i = steps[idx]
                            # 如果 mv_i 为 None，用 -1 表示特殊 token
                            mv_i_idx = self.tokenizer.encode_move(mv_i)
                            state_tensor = self.tokenizer.encode_state(s6x9_i)
                            # 前54个位置为状态表示
                            src_seq[i, :54] = state_tensor
                            # 第55个位置记录该步的 move 索引（如果有的话）
                            src_seq[i, 54] = mv_i_idx

                        # 假设上面已经生成了 src_seq，形状 (seq_len, 55)，其中 seq_len = self.history_len + 1
                        tgt_seq = torch.empty(seq_len, dtype=torch.long)
                        # 对于前 seq_len-1 个位置，用 "下一行的动作" 填充
                        for i in range(seq_len - 1):
                            # src_seq[i+1, 54] 就是下一时刻的动作
                            tgt_seq[i] = src_seq[i + 1, 54]
                        # 最后一项用 EOS
                        tgt_seq[seq_len - 1] = self.EOS_token

                        self.samples.append((src_seq, tgt_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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

# 测试用的main函数
if __name__ == '__main__':
    # 指定数据目录为当前目录下的 "rubik_shards"
    data_dir = 'rubik_shards'
    dataset = RubikDatasetPACT(data_dir=data_dir)
    print(f"从目录 {data_dir} 加载数据，共 {len(dataset)} 条样本。")