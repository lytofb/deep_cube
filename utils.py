# utils.py
import random

import torch
import pycuber as pc
from torch import nn
import torch.nn.functional as F

# 常见颜色，如果需要可以加更多
COLOR_CHARS = ['w', 'g', 'r', 'b', 'o', 'y']
# 你可以自定义颜色顺序

# 这里我把代码的MOVES_POOL改成了19，应该改动代码的什么地方
# 19 种合法转动
MOVES_POOL = [
    'U', 'U\'', 'U2', 'D', 'D\'', 'D2',
    'L', 'L\'', 'L2', 'R', 'R\'', 'R2',
    'F', 'F\'', 'F2', 'B', 'B\'', 'B2'
]
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES_POOL)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}

PAD_TOKEN = 18
SOS_TOKEN = 20
EOS_TOKEN = 19
MASK_OR_NOMOVE_TOKEN = 21

VOCAB_SIZE = 22  # 0..17(动作) + 18(PAD) + 19(MASK) + 20(EOS) + 21(SOS)

def convert_state_to_tensor(state_6x9, color_to_id=None):
    """
    state_6x9: 形如 [[c1..c9], [c1..c9], ..., 共6行], 每行9个字符
    color_to_id: dict, 把 'W','G','R','B','O','Y' 等映射到 0..5
    返回: 形如 (6*9,) 的 LongTensor 或 (6*9) 维embedding索引
    """
    if color_to_id is None:
        color_to_id = {c: i for i, c in enumerate(COLOR_CHARS)}

    flat = []
    for face_row in state_6x9:  # face_row is 9-length
        for color_char in face_row:
            # 有些数据里可能是小写 'r', 你可统一处理 to upper()
            # c = color_char.upper()
            if color_char not in color_to_id:
                # 如果遇到未知颜色，可以抛异常或映射到某个UNK
                raise ValueError(f"未知颜色: {color_char}")
            flat.append(color_to_id[color_char])
    # flat 长度是 6*9=54
    return torch.tensor(flat, dtype=torch.long)

def cube_to_6x9(cube):
    """
    将当前魔方 cube 序列化成 6x9 的二维数组, 去掉 '[ ]'。
    """
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    res = []
    for face in face_order:
        face_obj = cube.get_face(face)
        row_data = []
        for r in range(3):
            for c in range(3):
                raw_sticker = str(face_obj[r][c])  # e.g. "[g]"
                color_char = raw_sticker.strip('[]')
                row_data.append(color_char)
        res.append(row_data)
    return res

def move_str_to_idx(move_str):
    """把动作字符串 (如 'R','R2','R'') -> 0..17 的整数标签"""
    if move_str not in MOVE_TO_IDX:
        return MASK_OR_NOMOVE_TOKEN
        # raise ValueError(f"未知动作: {move_str}")
    return MOVE_TO_IDX[move_str]


def move_idx_to_str(move_idx):
    """把 0..17 -> 'R','R2','R'','U'..."""
    return IDX_TO_MOVE[move_idx]

def convert_tensor_to_state_6x9(tensor_54, id_to_color=None):
    """
    将形如 (54,) 的 LongTensor 转回 6×9 的颜色字符矩阵。

    Args:
        tensor_54: 形如 (54,) 的张量，每个元素是 0..5 等等，对应某种颜色
        id_to_color: dict, 比如 {0:'W', 1:'G', 2:'R', 3:'B', 4:'O', 5:'Y'} 等

    Returns:
        state_6x9: list[list[str]]，共 6 行，每行 9 个颜色字符
                   例如 row0 = [ 'W','W','W','W','W','W','W','W','W' ] 表示 Up 面
    """
    if id_to_color is None:
        id_to_color = {i: c for i, c in enumerate(COLOR_CHARS)}
    tensor_54 = tensor_54.view(-1)  # 确保是一维
    assert tensor_54.size(0) == 54, "输入张量必须长度为 54"

    state_6x9 = []
    idx = 0
    for face_idx in range(6):
        row_colors = []
        for _ in range(9):
            color_id = tensor_54[idx].item()
            color_char = id_to_color[color_id]
            row_colors.append(color_char)
            idx += 1
        state_6x9.append(row_colors)
    return state_6x9


char_to_fullcolor = {
    'r': 'red',
    'g': 'green',
    'b': 'blue',
    'y': 'yellow',
    'w': 'white',
    'o': 'orange',
    'u': 'unknown'
}


def create_cube_from_6x9(state_6x9):
    """
    根据 6×9 颜色字符布局，构造一个 pycuber.Cube 实例。

    假设 6×9 的行顺序分别对应 (U, L, F, R, B, D) 六个面，
    且每行的 9 个字符按 row-major (3×3) 顺序排列。
    """
    cube = pc.Cube()
    faces_order = ['U', 'L', 'F', 'R', 'B', 'D']
    for face_idx, face_name in enumerate(faces_order):
        face_str = state_6x9[face_idx]  # 这一行包含该面 9 个颜色字符
        face_obj = cube.get_face(face_name)
        idx = 0
        for row in range(3):
            for col in range(3):
                face_obj[row][col].colour = char_to_fullcolor[face_str[idx]]
                idx += 1
    return cube


def random_scramble_cube(steps=20):
    """随机打乱一个魔方并返回 (cube, moves)"""
    moves = [random.choice(MOVES_POOL) for _ in range(steps)]
    c = pc.Cube()
    for mv in moves:
        c(mv)
    return c, moves

class FocalLoss(nn.Module):
    """
    Focal Loss，用于缓解类别不平衡。
    loss = (1 - p_t)^gamma * CE(p_t)
    支持 ignore_index 来忽略 padding 位置。
    """
    def __init__(self, gamma: float = 2.0, ignore_index: int = None, reduction: str = 'mean'):
        """
        Args:
          gamma: 聚焦参数，gamma >= 0。gamma 越大，对简单样本的抑制越强。
          ignore_index: 在 target 中，等于该值的位置不参与损失计算。
          reduction: 'mean' 或 'sum' 或 'none'。
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logits: (B, C) 或 (N, C) 预测分数
          target: (B,) 或 (N,) 真实标签，标签值可为 0…C-1，也可为 ignore_index
        Returns:
          标量损失（如果 reduction!='none'）或与 target 同形的损失张量
        """
        # 1) 先计算每样本的交叉熵（不 reduction）
        ce_loss = F.cross_entropy(
            logits,
            target,
            reduction='none',
            ignore_index=self.ignore_index
        )  # (N,)

        # 2) 计算 p_t = exp(-CE)
        p_t = torch.exp(-ce_loss)

        # 3) 计算 Focal Loss
        loss = (1 - p_t) ** self.gamma * ce_loss  # (N,)

        # 4) 如果指定了 ignore_index，要把对应位置移除
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            loss = loss[valid_mask]

        # 5) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

if __name__ == '__main__':
    print(MOVE_TO_IDX)
    # cube,moves = random_scramble_cube(5)
    cube = pc.Cube()
    print(cube)
    state_54 = cube_to_6x9(cube)
    print(state_54)
    cube2 = create_cube_from_6x9(state_54)
    print(cube2)
    # print(cube2==cube)
    # cube2("L")