# inference.py
import torch
import pycuber as pc
import random
# from models.model_transformer import RubikTransformer
from models.model_history_transformer import RubikHistoryTransformer
# or CNN/Transformer...
from utils import convert_state_to_tensor, move_idx_to_str, MOVES_POOL


def is_solved(cube):
    """
    判断cube是否复原
    这里简单判断6个面是否颜色统一
    """
    for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
        face = cube.get_face(face_name)
        colors = set(str(face[r][c]).strip('[]') for r in range(3) for c in range(3))
        if len(colors) != 1:
            return False
    return True


def random_scramble_cube(steps=20):
    """随机打乱一个魔方并返回 cube 对象"""
    moves = [random.choice(MOVES_POOL) for _ in range(steps)]
    c = pc.Cube()
    for mv in moves:
        c(mv)
    return c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载训练好的模型
    # model = RubikTransformer(num_layers=24)
    model = RubikHistoryTransformer(num_layers=24)
    model.load_state_dict(torch.load("rubik_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 2. 随机打乱一个魔方
    cube = random_scramble_cube(steps=20)

    # 3. 循环推理
    max_steps = 50
    for step in range(max_steps):
        # 判断是否已复原
        if is_solved(cube):
            print(f"在 {step} 步内成功复原!")
            break

        # 将当前魔方状态转成网络输入
        s6x9 = []
        for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
            face_obj = cube.get_face(face_name)
            row_data = []
            for r in range(3):
                for c in range(3):
                    color_char = str(face_obj[r][c]).strip('[]').upper()
                    row_data.append(color_char)
            s6x9.append(row_data)

        # state_tensor = convert_state_to_tensor(s6x9).unsqueeze(0).to(device)  # (1,54)
        # 先把 54 维 state + 1 维伪 move(-1)，拼成 55
        curr_state_54 = convert_state_to_tensor(s6x9)  # shape (54,)
        dummy_move = torch.tensor([-1], dtype=torch.long)  # shape (1,)
        combined_55 = torch.cat([curr_state_54, dummy_move], dim=0)  # shape (55,)

        # 再加 batch_dim=1, seq_len=1 => (1,1,55)
        combined_55 = combined_55.unsqueeze(0).unsqueeze(0)  # shape (1,1,55)
        combined_55 = combined_55.float().to(device)
        # 前向
        with torch.no_grad():
            # logits = model(state_tensor)
            logits = model(combined_55)
            pred = torch.argmax(logits, dim=1).item()  # 0..17
        move_str = move_idx_to_str(pred)

        # 输出一下动作
        print(f"Step {step}: {move_str}")

        # 在cube上执行
        cube(move_str)
    else:
        print(f"超过 {max_steps} 步仍未复原，放弃。")


if __name__ == "__main__":
    main()
