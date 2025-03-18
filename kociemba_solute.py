import random
import pickle
import pycuber as pc
import subprocess
import os

# 所有合法旋转操作 (18种)
MOVES_POOL = [
    'U', "U'", 'U2', 'D', "D'", 'D2',
    'L', "L'", 'L2', 'R', "R'", 'R2',
    'F', "F'", 'F2', 'B', "B'", 'B2'
]


def inverse_move(move_str):
    """
    给定一个字符串形式的转动 (例如 'R', "R'", 'R2' 等)，
    返回它的逆操作 (如 'R' -> "R'", "R'" -> 'R', 'R2' -> 'R2')
    """
    if move_str.endswith('2'):
        return move_str
    elif move_str.endswith("'"):
        return move_str[:-1]
    else:
        return move_str + "'"


def cube_to_6x9(cube):
    """
    将魔方状态转换为 6x9 的二维数组表示，用于可视化。
    顺序依次为：U, L, F, R, B, D
    """
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    res = []
    for face in face_order:
        face_obj = cube.get_face(face)
        row_data = []
        for r in range(3):
            for c in range(3):
                # 去除 pycuber 输出中的中括号
                sticker = str(face_obj[r][c]).strip('[]')
                row_data.append(sticker)
        res.append(row_data)
    return res


def cube_to_kociemba(cube):
    """
    将当前魔方状态转换成 kociemba 求解器需要的 54 字符串格式。
    输出顺序为：U1...U9, R1...R9, F1...F9, D1...D9, L1...L9, B1...B9，
    并且将颜色映射为面字母： 'y' -> 'U', 'g' -> 'F', 'r' -> 'R',
    'b' -> 'B', 'o' -> 'L', 'w' -> 'D'
    """
    order = ['U', 'R', 'F', 'D', 'L', 'B']
    color_to_face = {
        'y': 'U',
        'g': 'F',
        'r': 'R',
        'b': 'B',
        'o': 'L',
        'w': 'D'
    }
    cube_str = ""
    for face in order:
        face_obj = cube.get_face(face)
        for r in range(3):
            for c in range(3):
                sticker = str(face_obj[r][c]).strip('[]')
                if sticker not in color_to_face:
                    raise ValueError(f"未知的 sticker 颜色: {sticker}")
                cube_str += color_to_face[sticker]
    return cube_str


def kociemba_solver(cube_str):
    """
    调用外部 kociemba 求解器，输入 cube_str 为 54 字符串，
    返回求解序列字符串（各步之间以空格分隔）。
    请根据实际路径修改 solver_path
    """
    solver_path = os.path.expanduser(
        "~/Public/qugy_workspace/data_dir/deep_cube_github/kociemba/kociemba/ckociemba/bin/kociemba")
    try:
        result = subprocess.check_output([solver_path, cube_str])
        solution_str = result.decode('utf-8').strip()
        return solution_str
    except Exception as e:
        print("调用 kociemba 求解器时出错:", e)
        return None


def generate_single_case(min_scramble=8, max_scramble=25):
    """
    生成单条数据，包含：
      - scramble: 随机打乱的操作序列（字符串）
      - solution: kociemba 求解器返回的复原操作序列（字符串）
      - steps: [(6x9状态, 所用操作), ...]，从打乱态到复原态的过程
    """
    # 1. 创建初始复原魔方
    cube = pc.Cube()

    # 2. 生成随机打乱操作
    k = random.randint(min_scramble, max_scramble)
    scramble_ops = [random.choice(MOVES_POOL) for _ in range(k)]
    for move in scramble_ops:
        cube(move)
    scramble_str = " ".join(scramble_ops)

    # 3. 将打乱后的魔方状态转换为 kociemba 输入字符串
    cube_str = cube_to_kociemba(cube)

    # 4. 调用 kociemba 求解器获得复原操作序列
    solution_str = kociemba_solver(cube_str)
    if solution_str is None:
        print("求解器未返回结果，采用空序列作为解")
        solution_ops = []
    else:
        solution_ops = solution_str.split()

    # 5. 记录从打乱态到复原态的步骤（初始状态 + 每一步执行后的状态）
    steps = []
    steps.append((cube_to_6x9(cube), None))
    for move in solution_ops:
        cube(move)
        steps.append((cube_to_6x9(cube), move))

    data_item = {
        # 'scramble': scramble_str,
        # 'solution': solution_str,
        'steps': steps
    }
    # print(data_item)
    return data_item


def generate_dataset(n=100):
    """
    批量生成 n 条魔方数据，每条数据都利用 kociemba 求解器获得复原序列。
    返回一个列表，每项数据包含 scramble, solution 和 steps。
    """
    data_list = []
    for _ in range(n):
        item = generate_single_case(min_scramble=8, max_scramble=25)
        data_list.append(item)
    return data_list


if __name__ == "__main__":
    # 生成测试数据，例如生成 10000 条数据
    dataset = generate_dataset(n=10000)
    # 保存到硬盘（此处使用 pickle 保存）
    with open("rubik_shards/rubik_data.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(f"成功生成 {len(dataset)} 条数据并存储到 rubik_data.pkl")
