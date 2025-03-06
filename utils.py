# utils.py
import torch

from tokenizer.gnn_tokenizer import GNNTokenizer

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
MASK_TOKEN = 21

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


def build_cube_edge_index():
    edges = set()

    def global_index(face, r, c):
        """将 (face, row, col) 转换为全局节点编号（共 54 个节点）"""
        return face * 9 + r * 3 + c

    # 1. 添加每个面内部的四邻域边（无向，每条边只添加一次）
    for face in range(6):
        offset = face * 9
        for r in range(3):
            for c in range(3):
                idx = offset + r * 3 + c
                # 向右连接
                if c < 2:
                    neighbor = offset + r * 3 + (c + 1)
                    edges.add(tuple(sorted((idx, neighbor))))
                # 向下连接
                if r < 2:
                    neighbor = offset + (r + 1) * 3 + c
                    edges.add(tuple(sorted((idx, neighbor))))

    # 2. 添加跨面边界的连接（每条边只添加一次）
    # 约定面顺序：0: U, 1: L, 2: F, 3: R, 4: B, 5: D

    # 边 1: U–F
    for c in range(3):
        u_idx = global_index(0, 2, c)  # U 面下边：行2，列c
        f_idx = global_index(2, 0, c)  # F 面上边：行0，列c
        edges.add(tuple(sorted((u_idx, f_idx))))

    # 边 2: U–L
    for i, (r, c) in enumerate([(0, 0), (1, 0), (2, 0)]):
        u_idx = global_index(0, r, 0)  # U 面左边：列0，行r
        l_idx = global_index(1, 0, 2 - i)  # L 面上边：行0，列(2-i)（反序）
        edges.add(tuple(sorted((u_idx, l_idx))))

    # 边 3: U–R
    for i, (r, c) in enumerate([(0, 2), (1, 2), (2, 2)]):
        u_idx = global_index(0, r, 2)  # U 面右边：列2，行r
        r_idx = global_index(3, 0, i)  # R 面上边：行0，列i
        edges.add(tuple(sorted((u_idx, r_idx))))

    # 边 4: U–B
    for c in range(3):
        u_idx = global_index(0, 0, c)  # U 面上边：行0，列c
        b_idx = global_index(4, 0, 2 - c)  # B 面上边：行0，列(2-c)（反序）
        edges.add(tuple(sorted((u_idx, b_idx))))

    # 边 5: D–F
    for c in range(3):
        d_idx = global_index(5, 0, c)  # D 面上边：行0，列c
        f_idx = global_index(2, 2, c)  # F 面下边：行2，列c
        edges.add(tuple(sorted((d_idx, f_idx))))

    # 边 6: D–L
    for i, (r, c) in enumerate([(0, 0), (1, 0), (2, 0)]):
        d_idx = global_index(5, i, 0)  # D 面左边：列0，行i
        l_idx = global_index(1, 2, 2 - i)  # L 面下边：行2，列(2-i)（反序）
        edges.add(tuple(sorted((d_idx, l_idx))))

    # 边 7: D–R
    for i, (r, c) in enumerate([(0, 2), (1, 2), (2, 2)]):
        d_idx = global_index(5, i, 2)  # D 面右边：列2，行i
        r_idx = global_index(3, 2, i)  # R 面下边：行2，列i
        edges.add(tuple(sorted((d_idx, r_idx))))

    # 边 8: D–B
    for c in range(3):
        d_idx = global_index(5, 2, c)  # D 面下边：行2，列c
        b_idx = global_index(4, 2, 2 - c)  # B 面下边：行2，列(2-c)（反序）
        edges.add(tuple(sorted((d_idx, b_idx))))

    # 边 9: F–L
    for i in range(3):
        f_idx = global_index(2, i, 0)  # F 面左边：列0，行i
        l_idx = global_index(1, i, 2)  # L 面右边：列2，行i
        edges.add(tuple(sorted((f_idx, l_idx))))

    # 边 10: F–R
    for i in range(3):
        f_idx = global_index(2, i, 2)  # F 面右边：列2，行i
        r_idx = global_index(3, i, 0)  # R 面左边：列0，行i
        edges.add(tuple(sorted((f_idx, r_idx))))

    # 边 11: B–L
    for i in range(3):
        b_idx = global_index(4, i, 2)  # B 面右边：列2，行i
        l_idx = global_index(1, i, 0)  # L 面左边：列0，行i
        edges.add(tuple(sorted((b_idx, l_idx))))

    # 边 12: B–R
    for i in range(3):
        b_idx = global_index(4, i, 0)  # B 面左边：列0，行i
        r_idx = global_index(3, i, 2)  # R 面右边：列2，行i
        edges.add(tuple(sorted((b_idx, r_idx))))

    # 转换为 tensor，edge_index 的 shape 应为 (2, 108)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

def move_str_to_idx(move_str):
    """把动作字符串 (如 'R','R2','R'') -> 0..17 的整数标签"""
    if move_str not in MOVE_TO_IDX:
        return PAD_TOKEN
        # raise ValueError(f"未知动作: {move_str}")
    return MOVE_TO_IDX[move_str]


def move_idx_to_str(move_idx):
    """把 0..17 -> 'R','R2','R'','U'..."""
    return IDX_TO_MOVE[move_idx]


if __name__ == '__main__':
    # 正确设置 edge_index
    edge_index = build_cube_edge_index()
    print("edge_index shape:", edge_index.shape)  # 例如：torch.Size([2, 108])

    # 初始化 GNNTokenizer
    tokenizer = GNNTokenizer(edge_index=edge_index, num_colors=6, gnn_hidden=64, gnn_out=128, gnn_layers=2)

    # 构造一个魔方状态测试用例
    # 此处假设魔方状态以6x9矩阵表示，每个元素为颜色ID (0~5)
    # 例如，直接构造一个简单的状态，依次填充 0,1,2,3,4,5 共 54 个元素
    s6x9 = [i % 6 for i in range(54)]
    print("s6x9:", s6x9)

    # 使用 tokenizer 对魔方状态进行编码，得到状态 embedding
    state_embedding = tokenizer.encode_state(s6x9)
    print("state embedding shape:", state_embedding.shape)  # 预期输出: torch.Size([128])

    # 测试动作编码，假设动作字符串为 "U"
    move_str = "U"
    move_idx = tokenizer.encode_move(move_str)
    print("Encoded move index for '{}': {}".format(move_str, move_idx))
