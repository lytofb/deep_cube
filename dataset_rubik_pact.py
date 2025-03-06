# dataset_rubik_pact.py
import os
import pickle
import torch
from torch.utils.data import Dataset

from quick_solute_cube import generate_scramble_and_solution, cube_to_6x9
from tokenizer.tokenizer_rubik import RubikTokenizer
import pycuber as pc

from utils import EOS_TOKEN

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

    # def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
    #     """
    #     如果你有 generate_single_case，可以在这里得到 steps = [(s6x9, move), ...]。
    #     只做滑窗切分，记录原始数据，而不做 encode。
    #     """
    #     from dataset_rubik_seq import generate_single_case  # 如果有这个函数
    #     print(f"RubikDatasetPACT: 正在内存中生成 {num_samples} 条数据...")
    #     for _ in range(num_samples):
    #         item = generate_single_case(min_scramble, max_scramble)
    #         steps = item['steps']  # [(state, move), ...]
    #         if len(steps) < self.history_len + 1:
    #             continue
    #         for t in range(self.history_len, len(steps)):
    #             # 构造原始 src
    #             src_seq = steps[t - self.history_len : t + 1]  # [(s6x9_i, mv_i), ...]
    #
    #             # 构造原始 tgt
    #             # 注意，这里不再插入 SOS/ EOS，也不 encode move
    #             tgt_list = [mv for (_, mv) in steps[t:]]  # 当前时刻到结尾
    #             # 存储 (src_seq, tgt_list)
    #             self.samples.append((src_seq, tgt_list))
    #
    #     print(f"RubikDatasetPACT: 内存生成完毕，共生成 {len(self.samples)} 条数据.")

    def _load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_list = pickle.load(f)
                except EOFError:
                    break
                for item in data_list:
                    self.samples.append(item)

                    # steps = item['steps']  # [(s6x9_i, mv_i), ...]
                    # if len(steps) < self.history_len + 1:
                    #     continue
                    # for t in range(self.history_len, len(steps)):
                    #     src_seq = steps[t - self.history_len : t + 1]
                    #     tgt_list = [mv for (_, mv) in steps[t:]]
                    #     self.samples.append((src_seq, tgt_list))

    def _generate_single_case(self,min_scramble=3, max_scramble=25):
        """
        生成一条数据，并直接返回 encode 后的 steps:
        返回字典格式: {'encoded_steps': [encoded_step, ...]}
        其中每个 encoded_step 是形状 (55,) 的 LongTensor，
        前 54 个数字为 state token，最后一个数字为 move token（若无动作则为 -1）。
        """
        # 导入 tokenizer
        tokenizer = RubikTokenizer()

        # 1) 新魔方
        cube = pc.Cube()
        # 2) 打乱 + 逆操作
        scramble_ops, solution_ops = generate_scramble_and_solution(min_scramble, max_scramble)
        # 3) 应用打乱操作
        for move in scramble_ops:
            cube(move)
        # 4) 记录打乱态
        raw_steps = []
        raw_steps.append((cube_to_6x9(cube), None))
        # 5) 依次应用 solution_ops，并记录每一步状态及对应动作
        for mv in solution_ops:
            cube(mv)
            raw_steps.append((cube_to_6x9(cube), mv))

        # 6) 对每一步进行 encode：构造 (55,) 张量：前54为 state token, 最后一位为 move token
        encoded_steps = []
        for s6x9, mv in raw_steps:
            if s6x9 is None:
                encoded_state = torch.full((54,), -1, dtype=torch.long)
            else:
                # tokenizer.encode_state 返回一个 list/array，可以转为 tensor
                encoded_state = torch.tensor(tokenizer.encode_state(s6x9), dtype=torch.long)
            if mv is None:
                encoded_move = -1  # 表示无动作
            else:
                encoded_move = tokenizer.encode_move(mv)
            # 拼接 state 与 move，得到形状 (55,)
            encoded_step = torch.cat([encoded_state, torch.tensor([encoded_move], dtype=torch.long)], dim=0)
            encoded_steps.append(encoded_step)

        return {'encoded_steps': encoded_steps}

    def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
        """
        使用 generate_single_case 生成 raw 的 encode 后的 steps，
        然后以滑窗方式切分，直接生成 (src_tensor, tgt_tensor) 样本，不再需要后续在 collate_fn 中做 encode。

        每个样本：
          - src_tensor: LongTensor, shape = (history_len+1, 55)
          - tgt_tensor: LongTensor, shape = (history_len+1,)
             前 history_len 个 token 来自 src_tensor 后移一位的动作编码，最后一位置为 EOS_TOKEN
        """
        from dataset_rubik_seq import generate_single_case  # 使用修改后的 generate_single_case
        print(f"RubikDatasetPACT: 正在内存中生成 {num_samples} 条数据...")

        for _ in range(num_samples):
            item = self._generate_single_case(min_scramble, max_scramble)
            encoded_steps = item['encoded_steps']  # 列表中每个元素形状为 (55,)
            # 如果生成的 steps 数量不足一个滑窗，跳过该样本
            if len(encoded_steps) < self.history_len + 1:
                continue

            # 滑窗切分：从 history_len 到 len(encoded_steps)-1，每个滑窗长度为 history_len+1
            for t in range(self.history_len, len(encoded_steps)):
                window = encoded_steps[t - self.history_len: t + 1]
                src_tensor = torch.stack(window, dim=0)  # (history_len+1, 55)

                # 构造 tgt_tensor: 长度与 src_tensor 相同
                # tgt_tensor[0:history_len] 来自 src_tensor[1:, 54]，最后一个 token 置为 EOS_TOKEN
                L = src_tensor.size(0)
                tgt_tensor = torch.empty(L, dtype=torch.long)
                for i in range(L - 1):
                    tgt_tensor[i] = src_tensor[i + 1, 54]
                tgt_tensor[L - 1] = EOS_TOKEN

                # 存储预处理好的样本
                self.samples.append((src_tensor, tgt_tensor))

        print(f"RubikDatasetPACT: 内存生成完毕，共生成 {len(self.samples)} 条数据.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回的 src_seq 是一个 list，里面每个元素=(s6x9_str, move_str or None)
        返回的 tgt_seq 是一个 list，里面是 move_str or None
        """
        return self.samples[idx]

    def getall(self):
        return self.samples

if __name__ == "__main__":
    # 测试：生成100条数据
    datasetPact = RubikDatasetPACT(data_dir=None,num_samples=1000)
    dataset = datasetPact.getall()

    # 保存到硬盘 (pickle 仅示例用，万亿级可能需要分块+多进程+分布式)
    import time
    timestamp_suffix = str(int(time.time()))[-6:]
    pid = os.getpid()  # 获取当前进程ID
    with open(f"rubik_shards_nocollate/rubik_data_{timestamp_suffix}_{pid}.pkl", "wb") as f:
      pickle.dump(dataset, f)

    print(f"成功生成 {len(dataset)} 条数据并存储到 rubik_data.pkl")

#bash
# for i in {1..20};do
# 	for i in {1..5}; do
# 	  python dataset_rubik_pact.py &
# 	done
# 	wait
# done
