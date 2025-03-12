import os
import pickle
import time
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from utils import convert_state_to_tensor, move_str_to_idx
from dataset_rubik_seq import generate_single_case
from tokenizer.tokenizer_rubik import RubikTokenizer

class RubikDataset(Dataset):
    """
    带历史的Dataset:
      输入：滑动窗口内连续的 (state, move) 对，共 (history_len+1, 55) 维张量，
            每行前54个位置为状态编码，第55个位置为对应 move 编码
      输出（目标序列）：以 SOS_token 开始，后续依次为从当前时刻到解法结束的动作编码，
            最后追加 EOS_token。即 tgt_seq = [SOS, move[t], move[t+1], ..., EOS]
    """
    def __init__(self, data_dir='data', history_len=8, max_files=None,
                 num_samples=0, min_scramble=8, max_scramble=25):
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
        print(f"RubikDataset: 正在内存中生成 {num_samples} 条数据...")
        for _ in range(num_samples):
            # 生成单条数据
            single_item = generate_single_case(min_scramble, max_scramble)
            steps = single_item['steps']
            # 若总步数不足 history_len+1，则跳过
            if len(steps) < self.history_len + 1:
                continue
            # 对每个解法序列，采用滑动窗口生成多个样本
            for t in range(self.history_len, len(steps)):
                seq_len = self.history_len + 1
                src_seq = torch.empty((seq_len, 55), dtype=torch.long)
                for i, idx in enumerate(range(t - self.history_len, t + 1)):
                    s6x9_i, mv_i = steps[idx]
                    # 如果 mv_i 为 None，用 -1 表示特殊 token
                    mv_i_idx = self.tokenizer.encode_move(mv_i)
                    state_tensor = self.tokenizer.encode_state(s6x9_i)
                    src_seq[i, :54] = state_tensor
                    src_seq[i, 54] = mv_i_idx

                # 构造 tgt: 以 SOS 为起始符，后面接上从当前时刻 t 开始到解法结束的所有 move，
                # 最后追加 EOS
                tgt_list = [self.SOS_token]
                for idx in range(t, len(steps)):
                    _, mv = steps[idx]
                    move_idx = self.tokenizer.encode_move(mv)
                    tgt_list.append(move_idx)
                tgt_list.append(self.EOS_token)
                tgt_seq = torch.tensor(tgt_list, dtype=torch.long)

                self.samples.append((src_seq, tgt_seq))
        print(f"RubikDataset: 内存生成完毕，共生成 {len(self.samples)} 条数据.")

    def _load_from_file(self, file_path):
        """
        加载预处理好的数据文件，文件中存储的直接就是 (src_seq, tgt_seq) 格式的样本，
        无需再做额外处理。
        """
        with open(file_path, 'rb') as f:
            samples = pickle.load(f)
            self.samples.extend(samples)

    def getall(self):
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """
    batch: list of (src_seq, tgt_seq)
      - src_seq.shape = (history_len+1, 55)
      - tgt_seq: 1D tensor, 长度可能不同
    """
    src_seqs = [x[0] for x in batch]
    tgt_seqs = [x[1] for x in batch]

    # 固定长度的 src 直接 stack
    src_tensor = torch.stack(src_seqs, dim=0)  # (B, history_len+1, 55)

    # 对 tgt_seqs 进行 padding，使它们具有相同长度
    tgt_tensor = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_TOKEN)

    return src_tensor, tgt_tensor

if __name__ == "__main__":
    # 测试：生成1000条数据
    dataset = RubikDataset(data_dir=None, num_samples=1000)
    all_samples = dataset.getall()

    # 保存到硬盘 (pickle 仅示例用，实际大规模数据可能需要分块处理)
    timestamp_suffix = str(int(time.time()))[-6:]
    pid = os.getpid()  # 获取当前进程ID
    output_dir = "rubik_shards_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rubik_data_{timestamp_suffix}_{pid}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"成功生成 {len(all_samples)} 条数据并存储到 {output_path}")
