# utils.py
import torch

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


def move_str_to_idx(move_str):
    """把动作字符串 (如 'R','R2','R'') -> 0..17 的整数标签"""
    if move_str not in MOVE_TO_IDX:
        return PAD_TOKEN
        # raise ValueError(f"未知动作: {move_str}")
    return MOVE_TO_IDX[move_str]


def move_idx_to_str(move_idx):
    """把 0..17 -> 'R','R2','R'','U'..."""
    return IDX_TO_MOVE[move_idx]
