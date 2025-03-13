import math,torch
from utils import convert_state_to_tensor, PAD_TOKEN, move_str_to_idx, move_idx_to_str


# --- 修改后的 CubeTokenizer 类 ---
class CubeTokenizer:
    def __init__(self):
        pass

    def permutation_to_index(self, perm):
        rank = 0
        n = len(perm)
        for i in range(n):
            if perm[i] is None:
                raise ValueError(f"Permutation element at index {i} is None: {perm}")
            smaller = 0
            for j in range(i + 1, n):
                if perm[j] is None:
                    raise ValueError(f"Permutation element at index {j} is None: {perm}")
                if perm[j] < perm[i]:
                    smaller += 1
            rank += smaller * math.factorial(n - i - 1)
        return rank

    def orientation_to_index(self, orientations, base):
        index = 0
        n = len(orientations)
        for i in range(n - 1):
            if orientations[i] is None:
                raise ValueError(f"Orientation element at index {i} is None: {orientations}")
            index = index * base + orientations[i]
        return index

    # --- 新增辅助函数 ---
    def int_to_base6(self, number, length):
        """
        将整数 number 转换为固定长度 length 的 base-6 数字列表（高位在前）。
        如果 number 小于 6^(length)，高位补 0。
        """
        digits = [0] * length
        for i in range(length - 1, -1, -1):
            digits[i] = number % 6
            number //= 6
        return digits

    def state54_to_cubie_state_ULFRBD(self, state_faces):
        # 修改说明：假定 state_faces 顺序为 [U, L, F, R, B, D]
        flat = [color for face in state_faces for color in face]
        centers = {
            'U': flat[4],
            'L': flat[13],
            'F': flat[22],
            'R': flat[31],
            'B': flat[40],
            'D': flat[49],
        }
        cornerFacelet = [
            [8, 27, 20],   # URF
            [6, 18, 11],   # UFL
            [0, 9, 38],    # ULB
            [2, 36, 29],   # UBR
            [53, 26, 35],  # DFR
            [45, 17, 24],  # DLF
            [51, 44, 15],  # DBL
            [47, 33, 42],  # DRB
        ]
        edgeFacelet = [
            [5, 28],   # UR
            [7, 19],   # UF
            [3, 10],   # UL
            [1, 37],   # UB
            [50, 34],  # DR
            [46, 25],  # DF
            [48, 16],  # DL
            [52, 43],  # DB
            [23, 30],  # FR
            [21, 14],  # FL
            [12, 41],  # BL
            [42, 32],  # BR
        ]
        solvedCorners = [
            [centers['U'], centers['R'], centers['F']],  # URF
            [centers['U'], centers['F'], centers['L']],  # UFL
            [centers['U'], centers['L'], centers['B']],  # ULB
            [centers['U'], centers['B'], centers['R']],  # UBR
            [centers['D'], centers['F'], centers['R']],  # DFR
            [centers['D'], centers['L'], centers['F']],  # DLF
            [centers['D'], centers['B'], centers['L']],  # DBL
            [centers['D'], centers['R'], centers['B']],  # DRB
        ]
        solvedEdges = [
            [centers['U'], centers['R']],  # UR
            [centers['U'], centers['F']],  # UF
            [centers['U'], centers['L']],  # UL
            [centers['U'], centers['B']],  # UB
            [centers['D'], centers['R']],  # DR
            [centers['D'], centers['F']],  # DF
            [centers['D'], centers['L']],  # DL
            [centers['D'], centers['B']],  # DB
            [centers['F'], centers['R']],  # FR
            [centers['F'], centers['L']],  # FL
            [centers['B'], centers['L']],  # BL
            [centers['B'], centers['R']],  # BR
        ]
        corner_perm = [None] * 8
        corner_orient = [None] * 8
        for i in range(8):
            colors = [flat[idx] for idx in cornerFacelet[i]]
            found = False
            for j in range(8):
                solved = solvedCorners[j]
                if sorted(colors) == sorted(solved):
                    for ori in range(3):
                        rotated = colors[ori:] + colors[:ori]
                        if rotated[0] in (centers['U'], centers['D']):
                            corner_perm[i] = j
                            corner_orient[i] = ori % 3
                            found = True
                            break
                    if found:
                        break
            if not found:
                raise ValueError(f"未能匹配角块{i}的颜色: {colors}")
        edge_perm = [None] * 12
        edge_orient = [None] * 12
        for i in range(12):
            colors = [flat[idx] for idx in edgeFacelet[i]]
            found = False
            for j in range(12):
                solved = solvedEdges[j]
                if sorted(colors) == sorted(solved):
                    orient = 0 if colors[0] in (centers['U'], centers['D']) else 1
                    edge_perm[i] = j
                    edge_orient[i] = orient
                    found = True
                    break
            if not found:
                raise ValueError(f"未能匹配棱块{i}的颜色: {colors}")
        return {
            'corner_permutation': corner_perm,
            'corner_orientation': corner_orient,
            'edge_permutation': edge_perm,
            'edge_orientation': edge_orient
        }

    # --- 具体实现 encode_state  ---
    def encode_state(self, state_faces):
        """
        将输入的 54 贴纸状态（6×9，顺序：U, L, F, R, B, D）转换为一个 54 维的 token 向量，
        其中按照群的性质分别将：
          - 角块排列（CP）编码为 12 个 base-6 数字
          - 角块朝向（CO）编码为 9 个 base-6 数字
          - 棱块排列（EP）编码为 24 个 base-6 数字
          - 棱块朝向（EO）编码为 9 个 base-6 数字
        """
        # 1. 利用已有方法解析出 cubie state
        cubie_state = self.state54_to_cubie_state_ULFRBD(state_faces)
        # 2. 分别计算 4 个 token 的整数表示
        cp_index = self.permutation_to_index(cubie_state['corner_permutation'])   # 范围 0～40319
        co_index = self.orientation_to_index(cubie_state['corner_orientation'], base=3)  # 范围 0～2186
        ep_index = self.permutation_to_index(cubie_state['edge_permutation'])       # 范围 0～479001599
        eo_index = self.orientation_to_index(cubie_state['edge_orientation'], base=2)  # 范围 0～2047

        # 3. 将各 token 转换为固定长度的 base-6 数字列表
        cp_digits = self.int_to_base6(cp_index, 12)  # 12 位
        co_digits = self.int_to_base6(co_index, 9)   # 9 位
        ep_digits = self.int_to_base6(ep_index, 24)  # 24 位
        eo_digits = self.int_to_base6(eo_index, 9)   # 9 位

        # 4. 拼接成 54 维的序列
        tokens = cp_digits + co_digits + ep_digits + eo_digits  # 总长度 12+9+24+9 = 54
        return torch.tensor(tokens, dtype=torch.long)

    def encode_move(self, move):
        # 根据需求实现 move 的编码，此处仅为示例，可自行扩展
        # 原先是 move_str_to_idx(mv)，也可加更多逻辑
        if move is None:
            return PAD_TOKEN
        else:
            return move_str_to_idx(move)

    def decode_move(self, token):
        raise NotImplementedError("CubeTokenizer 暂不支持 move 的解码。")

# --- 修改后的 state54_to_cubie_tokens 函数 ---
def state54_to_cubie_tokens(state_faces):
    tokenizer = CubeTokenizer()
    tokens = tokenizer.encode_state(state_faces)
    return tokens



# 示例用法
if __name__ == "__main__":
    # 输入状态为6个面，每个面9个元素（顺序：U, R, F, D, L, B）
    # 帮我修改一下，顺序是ULFRBD
    # state = [
    #     ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
    #     ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'],
    #     ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
    #     ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
    #     ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'],
    #     ['y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']
    # ]
    state = [
        ['r', 'w', 'y', 'w', 'y', 'y', 'w', 'o', 'b'],
        ['y', 'g', 'g', 'y', 'r', 'r', 'w', 'r', 'g'],
        ['r', 'g', 'o', 'b', 'g', 'b', 'o', 'w', 'b'],
        ['w', 'o', 'o', 'y', 'o', 'o', 'y', 'r', 'o'],
        ['b', 'o', 'g', 'b', 'b', 'g', 'g', 'r', 'r'],
        ['w', 'b', 'r', 'y', 'w', 'g', 'b', 'w', 'y']
    ]
    try:
        tokens = state54_to_cubie_tokens(state)
        print("转换后得到的Token：")
        for key, value in tokens.items():
            print(f"{key}: {value}")
    except ValueError as e:
        print("转换错误：", e)
