import random
import os
import pickle
import redis

import torch
from torch.utils.data import Dataset, DataLoader

import pycuber as pc

# ------------------------------------------------------------------
# 一些全局定义 (包含 PAD/SOS/EOS 令牌)
# ------------------------------------------------------------------

MOVE_MAP = {
    'U': 0, 'U\'': 1, 'U2': 2,
    'D': 3, 'D\'': 4, 'D2': 5,
    'L': 6, 'L\'': 7, 'L2': 8,
    'R': 9, 'R\'': 10, 'R2': 11,
    'F': 12, 'F\'': 13, 'F2': 14,
    'B': 15, 'B\'': 16, 'B2': 17
}

from utils import (
    COLOR_CHARS,
    MOVES_POOL,
    MOVE_TO_IDX,
    PAD_TOKEN,
    MASK_OR_NOMOVE_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    VOCAB_SIZE,
    convert_state_to_tensor,
    move_str_to_idx,
)


def inverse_move(move_str):
    """给定一个转动(如 'R','R'','R2')，返回它的逆操作。"""
    if move_str.endswith('2'):
        return move_str
    elif move_str.endswith('\''):
        return move_str[:-1]
    else:
        return move_str + '\''


def generate_scramble_and_solution(min_scramble=3, max_scramble=25):
    """随机打乱 + 逆操作得到还原序列。"""
    k = random.randint(min_scramble, max_scramble)
    scramble_moves = [random.choice(MOVES_POOL) for _ in range(k)]
    # 逆序还原
    solution_moves = [inverse_move(m) for m in reversed(scramble_moves)]
    return scramble_moves, solution_moves


def cube_to_6x9(cube):
    """把魔方序列化成 6x9 的二维数组 (去除 '[ ]')。"""
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    res = []
    for face in face_order:
        face_obj = cube.get_face(face)
        row_data = []
        for r in range(3):
            for c in range(3):
                raw_sticker = str(face_obj[r][c])  # "[g]"之类
                color_char = raw_sticker.strip('[]')
                row_data.append(color_char)
        res.append(row_data)
    return res


def generate_single_case(min_scramble=3, max_scramble=25):
    """
    生成一条数据: steps = [ (6x9二维数组, move), ... ]
    steps[0] 为打乱态 (move=None)
    后续 steps[i] 为执行第i步操作后的状态
    """
    # 1) 新魔方
    cube = pc.Cube()
    # 2) 打乱 + 逆操作
    scramble_ops, solution_ops = generate_scramble_and_solution(min_scramble, max_scramble)
    # 3) 应用打乱操作
    for move in scramble_ops:
        cube(move)
    # 4) 记录打乱态
    steps = []
    steps.append((cube_to_6x9(cube), None))
    # 5) 依次应用 solution_ops
    for mv in solution_ops:
        cube(mv)
        steps.append((cube_to_6x9(cube), mv))
    return {'steps': steps}



def moves_to_action_ids(move_list):
    """
    将字符串 moves 映射成 action id
    """
    action_ids = []
    for mv in move_list:
        if mv is None:
            continue
        if mv in MOVE_MAP:
            action_ids.append(MOVE_MAP[mv])
        else:
            # 未知动作
            action_ids.append(-1)
    return torch.tensor(action_ids, dtype=torch.long)


# ------------------------------------------------------------------
# 改造后的 RubikSeqDataset，支持 Redis 缓存
# ------------------------------------------------------------------
class RubikSeqDataset(Dataset):
    """
    根据给定的打乱/还原逻辑生成数据，但返回格式与第一个类保持一致:
      (cond(54,), seq) => seq中含 [SOS_TOKEN] 和 [EOS_TOKEN].

    新增参数:
      use_redis: 是否使用 Redis 缓存数据 (默认为 False)
      redis_host, redis_port, redis_db: Redis 连接参数
      redis_key: 存储数据的 Redis key
    """

    def __init__(self,
                 pkl_file=None,
                 num_samples=0,
                 min_scramble=8,
                 max_scramble=25,
                 use_redis=False,
                 redis_host='localhost',
                 redis_port=6379,
                 redis_db=0,
                 redis_key='rubik_seq_dataset'):
        super().__init__()
        self.data_list = []
        self.use_redis = use_redis
        self.redis_key = redis_key

        if self.use_redis:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            # 尝试从 Redis 中加载数据
            cached_data = self.redis_client.get(self.redis_key)
            if cached_data is not None:
                self.data_list = pickle.loads(cached_data)
                print(f"RubikSeqDataset: 从 Redis key '{self.redis_key}' 加载了 {len(self.data_list)} 条数据")
            else:
                print(f"RubikSeqDataset: Redis 中未找到 key '{self.redis_key}'，开始加载/生成数据...")
                if pkl_file is not None:
                    if os.path.isdir(pkl_file):
                        self._load_from_pickle_dir(pkl_file)
                    else:
                        raise ValueError(f"pkl_file 参数必须为包含 pkl 文件的目录，而不是文件: {pkl_file}")
                else:
                    self._generate_in_memory(num_samples, min_scramble, max_scramble)
                # 将加载/生成的数据存入 Redis
                self.redis_client.set(self.redis_key, pickle.dumps(self.data_list))
                print(f"RubikSeqDataset: 已将 {len(self.data_list)} 条数据存入 Redis key '{self.redis_key}'")
        else:
            if pkl_file is not None:
                if os.path.isdir(pkl_file):
                    self._load_from_pickle_dir(pkl_file)
                else:
                    raise ValueError(f"pkl_file 参数必须为包含 pkl 文件的目录，而不是文件: {pkl_file}")
            else:
                self._generate_in_memory(num_samples, min_scramble, max_scramble)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        返回与第一个类同样的结构:
          (cond(54,), seq张量)  其中 seq = [SOS_TOKEN, ..., EOS_TOKEN].
        """
        item = self.data_list[idx]
        steps = item['steps']  # [(6x9, move), ...]

        # 取初始打乱态 (6x9) => 映射到 (54,)
        scrambled_6x9 = steps[0][0]
        cond = convert_state_to_tensor(scrambled_6x9)

        # 解法动作 => steps[1..]
        solution_ops = [st[1] for st in steps[1:]]  # list of str
        action_ids = moves_to_action_ids(solution_ops)

        # 在动作序列前后添加 [SOS_TOKEN], [EOS_TOKEN]
        seq_ids = [SOS_TOKEN] + action_ids.tolist() + [EOS_TOKEN]
        seq_tensor = torch.tensor(seq_ids, dtype=torch.long)

        return cond, seq_tensor

    def _load_from_pickle(self, pkl_file):
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"pkl_file={pkl_file} 不存在")
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        print(f"RubikSeqDataset: 从 {pkl_file} 加载了 {len(self.data_list)} 条数据")

    def _generate_in_memory(self, num_samples, min_scramble, max_scramble):
        print(f"RubikSeqDataset: 正在内存中生成 {num_samples} 条数据...")
        for _ in range(num_samples):
            # 生成单条数据
            single_item = generate_single_case(min_scramble, max_scramble)
            self.data_list.append(single_item)
        print(f"RubikSeqDataset: 内存生成完毕，共生成 {len(self.data_list)} 条数据.")

# 新增方法
def _load_from_pickle_dir(self, pkl_dir):
    if not os.path.exists(pkl_dir):
         raise FileNotFoundError(f"目录 {pkl_dir} 不存在")
    pkl_files = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    if not pkl_files:
         raise FileNotFoundError(f"在目录 {pkl_dir} 中未找到任何 pkl 文件")
    for file in pkl_files:
         with open(file, 'rb') as f:
              data = pickle.load(f)
              # 如果 data 是列表，则合并所有数据；否则直接添加
              if isinstance(data, list):
                  self.data_list.extend(data)
              else:
                  self.data_list.append(data)
    print(f"RubikSeqDataset: 从目录 {pkl_dir} 加载了 {len(self.data_list)} 条数据")

# ------------------------------------------------------------------
# 与第一个类的 collate_fn 保持一致的拼接逻辑
# ------------------------------------------------------------------
def collate_fn(batch):
    """
    对齐变长序列, 以 PAD_TOKEN=18 补齐.
    返回:
      conds: (B,54)
      padded_seqs: (B, max_len)
      lengths: list[int]，表示每条序列的原始长度
    """
    conds = []
    seqs_list = []
    for cond, seq in batch:
        conds.append(cond)
        seqs_list.append(seq)

    conds = torch.stack(conds, dim=0)  # (B,54)
    lengths = [s.size(0) for s in seqs_list]
    max_len = max(lengths)

    padded_seqs = torch.full((len(batch), max_len),
                             fill_value=PAD_TOKEN,
                             dtype=torch.long)
    for i, s in enumerate(seqs_list):
        padded_seqs[i, :s.size(0)] = s

    return conds, padded_seqs, lengths


# ------------------------------------------------------------------
# 小测试
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 测试在内存生成并使用 Redis 缓存
    ds_inmem = RubikSeqDataset(num_samples=10000, min_scramble=3, max_scramble=5, use_redis=True,
                               redis_host='localhost', redis_port=6379, redis_db=0, redis_key='rubik_seq_dataset_test')
    print("Dataset size:", len(ds_inmem))

    dl = DataLoader(ds_inmem, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for conds, seqs, lengths in dl:
        print("conds.shape =", conds.shape)  # (B,54)
        print("seqs.shape =", seqs.shape)    # (B, max_seq_len)
        print("lengths =", lengths)          # 每条的原始序列长度
        print("seqs =", seqs)
        break
