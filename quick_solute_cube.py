import random
import pickle
import pycuber as pc

# 所有合法旋转操作 (18种)
MOVES_POOL = [
    'U', 'U\'', 'U2', 'D', 'D\'', 'D2',
    'L', 'L\'', 'L2', 'R', 'R\'', 'R2',
    'F', 'F\'', 'F2', 'B', 'B\'', 'B2'
]


def inverse_move(move_str):
    """
    给定一个字符串形式的转动 (例如 'R', 'R'', 'R2' 等)，
    返回它的逆操作 (如 'R'->'R'', 'R''->'R', 'R2'->'R2')。
    """
    # 简单处理三种情况
    if move_str.endswith('2'):
        # 'R2' 的逆操作依然是 'R2'
        return move_str
    elif move_str.endswith('\''):
        # 'R'' 的逆操作是 'R'
        return move_str[:-1]  # 去掉尾部 '
    else:
        # 'R' 的逆操作是 "R'"
        return move_str + '\''


def generate_scramble_and_solution(min_scramble=1, max_scramble=25):
    """
    随机生成打乱和对应的逆序还原操作序列 (不调用任何求解器)。
    返回 (scramble_moves_list, solution_moves_list)
      - scramble_moves_list: 正向打乱的操作列表(字符串)
      - solution_moves_list: 逆向还原的操作列表(字符串)
    """
    k = random.randint(min_scramble, max_scramble)
    scramble_moves = [random.choice(MOVES_POOL) for _ in range(k)]

    # 构造逆序还原操作
    # 先将 scramble_moves 反转，再对每个 move 取逆
    solution_moves = [inverse_move(m) for m in reversed(scramble_moves)]

    return scramble_moves, solution_moves


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


def generate_single_case(min_scramble=3, max_scramble=25):
    """
    生成单条数据，包含:
      - scramble_str
      - solution_str
      - steps: [(6x9状态, 所用操作), ...] (从打乱态到复原态)
    """
    # 1. 创建复原魔方
    cube = pc.Cube()

    # 2. 生成打乱操作、求解操作
    scramble_ops, solution_ops = generate_scramble_and_solution(min_scramble, max_scramble)

    # 3. 正向应用 scramble_ops 得到杂乱态
    for move in scramble_ops:
        cube(move)

    # 4. 记录 "从打乱态 -> 复原态" 的全过程
    #    初始状态(杂乱态) + 每一步应用 solution_ops 之后的状态
    steps = []

    # (a) 依次应用 solution_ops, 并记录
    for move in solution_ops:
        steps.append((cube_to_6x9(cube), move))
        cube(move)

    # (b) 最终状态
    steps.append((cube_to_6x9(cube), None))


    data_item = {
        'steps': steps  # [(6x9二维数组, move or None), ...]
    }
    return data_item


def generate_dataset(n=100):
    """
    批量生成 n 条魔方数据 (无求解器)。返回一个列表。
    """
    data_list = []
    for _ in range(n):
        item = generate_single_case(min_scramble=8, max_scramble=25)
        data_list.append(item)
    return data_list


if __name__ == "__main__":
    # 测试：生成100条数据
    dataset = generate_dataset(n=10000)

    # 保存到硬盘 (pickle 仅示例用，万亿级可能需要分块+多进程+分布式)
    with open("rubik_shards/rubik_data.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"成功生成 {len(dataset)} 条数据并存储到 rubik_data.pkl")
