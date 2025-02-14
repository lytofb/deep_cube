import random
import pickle
import pycuber as pc
from pycuber.solver import CFOPSolver


def cube_to_6x9array(cube):
    """
    将一个 pycuber.Cube 对象转换为 shape=(6, 9) 的二维列表，
    并去掉贴纸字符串中的方括号 [ ]。
    """
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    array_6x9 = []
    for face_name in face_order:
        face_obj = cube.get_face(face_name)
        row_data = []
        for r in range(3):
            for c in range(3):
                raw_sticker = str(face_obj[r][c])  # 例如 "[g]"、"[r]" 等
                color_char = raw_sticker.strip('[]')  # 去掉首尾方括号
                row_data.append(color_char)
        array_6x9.append(row_data)
    return array_6x9


def generate_scrambled_cube(min_scramble=1, max_scramble=25):
    """
    生成一个随机打乱的魔方对象，并返回 (打乱后的魔方, 打乱公式)。
    打乱步数范围在 [min_scramble, max_scramble] 间随机。
    """
    moves_pool = [
        'U', 'U\'', 'U2', 'D', 'D\'', 'D2',
        'L', 'L\'', 'L2', 'R', 'R\'', 'R2',
        'F', 'F\'', 'F2', 'B', 'B\'', 'B2'
    ]
    scramble_length = random.randint(min_scramble, max_scramble)

    random_moves = [random.choice(moves_pool) for _ in range(scramble_length)]
    scramble_formula = pc.Formula(random_moves)

    # 创建一个复原魔方，然后应用打乱公式
    cube = pc.Cube()
    cube(scramble_formula)

    return cube, scramble_formula


def solve_and_get_steps_as_6x9(cube):
    """
    使用 CFOPSolver 对打乱后的魔方进行求解，返回：
      - states: 从打乱状态到复原状态的所有中间状态(6x9数组)
      - moves:  对应每一步操作(这里会是字符串)
    """
    tmp_cube = cube.copy()
    solver = CFOPSolver(tmp_cube)
    solution_moves = solver.solve()  # ['R', 'U', 'R2', ...]

    states = []
    moves = []

    # 初始打乱状态(尚未执行任何操作)
    states.append(cube_to_6x9array(cube))
    moves.append(None)  # 第一条没有操作

    # 依次执行操作，并记录
    tmp_cube = cube.copy()
    for move in solution_moves:
        tmp_cube(move)
        states.append(cube_to_6x9array(tmp_cube))
        # 这里将 move 转成字符串
        moves.append(str(move))

    return states, moves


def generate_rubik_data(n=1):
    """
    生成 n 个打乱-求解数据，每条数据包含:
      1. scramble: 打乱公式字符串
      2. solution: 求解公式字符串
      3. steps: [(6x9数组, move_string), ...]
    """
    all_data = []
    for i in range(n):
        # 生成随机打乱的魔方 (1~25 步)
        scrambled_cube, scramble_formula = generate_scrambled_cube(min_scramble=1, max_scramble=25)

        # 获取求解过程
        states, moves = solve_and_get_steps_as_6x9(scrambled_cube)

        # 转换打乱公式、求解公式为字符串
        scramble_str = ' '.join(str(m) for m in scramble_formula)
        # 注意：moves[0] = None，后面都是字符串，所以可以安全跳过
        solution_str = ' '.join(m for m in moves[1:] if m)

        # 组合步骤数据
        steps_data = list(zip(states, moves))

        data_item = {
            "scramble": scramble_str,
            "solution": solution_str,
            "steps": steps_data  # steps_data 里包含 (list_of_list_6x9, move_str or None)
        }
        all_data.append(data_item)

    return all_data


if __name__ == "__main__":
    # 生成 10 条数据用于演示
    rubik_data_list = generate_rubik_data(n=10)

    # 尝试保存到本地 (pickle)
    filename = "rubik_shards/rubik_data.pkl"
    with open(filename, "wb") as f:
        pickle.dump(rubik_data_list, f)
    print(f"数据已保存到 {filename}。")

    # 再尝试读取回来看是否成功
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    print(f"从 {filename} 成功加载了 {len(loaded_data)} 条数据。")
