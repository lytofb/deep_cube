# inference_seq2seq.py

import torch
import random
import pycuber as pc

# 注意：从你的 seq2seq 模型文件导入
from models.model_history_transformer import RubikSeq2SeqTransformer

from utils import (
    convert_state_to_tensor,  # 需兼容6x9输入 => (54,)
    MOVES_POOL,
    move_idx_to_str,
    PAD_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN
)

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

def is_solved(cube):
    """判断 cube 是否复原：6 个面是否颜色统一"""
    for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
        face = cube.get_face(face_name)
        colors = set(str(face[r][c]).strip('[]') for r in range(3) for c in range(3))
        if len(colors) != 1:
            return False
    return True

def random_scramble_cube(steps=20):
    """随机打乱一个魔方并返回 (cube, moves)"""
    moves = [random.choice(MOVES_POOL) for _ in range(steps)]
    c = pc.Cube()
    for mv in moves:
        c(mv)
    return c, moves

def build_src_tensor_from_cube(cube):
    """
    构造模型需要的 src: (1, 1, 55)
    - 前 54 维是魔方当前状态
    - 最后 1 维放一个占位 move，比如 PAD_TOKEN
    """
    # 按你训练时的方式获取 (6x9) 的颜色字符数组
    s6x9 = []
    for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
        face_obj = cube.get_face(face_name)
        row_data = []
        for r in range(3):
            for c in range(3):
                color_char = str(face_obj[r][c]).strip('[]').lower()
                row_data.append(color_char)
        s6x9.append(row_data)

    # 转成 (54,) 的张量
    state_54 = convert_state_to_tensor(s6x9)  # (54,)

    # 末尾补一个占位 move（训练时第55维存 move 索引）
    dummy_move = torch.tensor([PAD_TOKEN], dtype=torch.long)  # shape (1,)

    combined_55 = torch.cat([state_54, dummy_move], dim=0)  # => (55,)

    # 加上 batch=1, seq_len=1 => (1,1,55)
    combined_55 = combined_55.unsqueeze(0).unsqueeze(0)
    return combined_55  # (1,1,55)

def greedy_decode_seq2seq(model, src, max_len=50):
    """
    对单条样本(batch=1)使用贪心解码产生动作序列（不含 SOS/EOS）。
    - src: (1, src_seq_len, 55)，本例中 src_seq_len=1
    - 返回: List[int]，预测出的 move token 序列
    """
    device = src.device
    model.eval()

    # 初始 Decoder 输入: [SOS_TOKEN], shape=(1,1)
    decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)

    predicted_tokens = []
    with torch.no_grad():
        for _ in range(max_len):
            # 前向: model(src, decoder_input)
            logits = model(src, decoder_input)
            # logits => (1, decoder_input_len, num_moves)

            # 取最后时刻的输出 => shape (1, num_moves)
            last_step_logits = logits[:, -1, :]
            next_token = torch.argmax(last_step_logits, dim=1)  # => (1,)

            if next_token.item() == PAD_TOKEN:
                # 说明模型输出的是 PAD，用来占位
                # 你可以选择 break（终止解码）或 continue（直接跳过）
                break
            elif next_token.item() == EOS_TOKEN:
                break

            # 否则，拼到 decoder_input
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            predicted_tokens.append(next_token.item())

    return predicted_tokens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载训练好的 seq2seq 模型，使用 config 中的参数
    model = RubikSeq2SeqTransformer(
        input_dim=config.model.input_dim,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        nhead=config.model.nhead,
        num_moves=config.model.num_moves,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    model.load_state_dict(torch.load(config.inference.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 随机打乱魔方
    scramble_steps = config.inference.scramble_steps
    cube, scramble_moves = random_scramble_cube(steps=scramble_steps)
    print(f"Scramble moves: {scramble_moves}")

    # 3. 构造 src
    src_tensor = build_src_tensor_from_cube(cube).to(device)
    # 4. 用 seq2seq 进行贪心解码，生成还原动作序列
    pred_tokens = greedy_decode_seq2seq(model, src_tensor, max_len=config.inference.max_len)
    print(f"Predicted tokens: {pred_tokens}")
    # 转成字符串动作
    pred_moves = []
    for t in pred_tokens:
        if t == PAD_TOKEN:
            # 不转换成动作，可能直接 break / 跳过
            break
        elif t == EOS_TOKEN:
            break
        else:
            pred_moves.append(move_idx_to_str(t))
    # pred_moves = [move_idx_to_str(t) for t in pred_tokens]
    print(f"Predicted moves: {pred_moves}")

    # 5. 依次执行预测动作，检查能否复原
    max_steps = len(pred_moves)
    for i, mv in enumerate(pred_moves):
        cube(mv)
        if is_solved(cube):
            print(f"在第 {i+1} 步成功复原!")
            break
    else:
        print(f"执行完 {max_steps} 步也未复原，可以尝试改进模型或延长解码。")

if __name__ == "__main__":
    main()
