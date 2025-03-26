# inference_seq2seq.py
from collections import OrderedDict

import torch
import random
import pycuber as pc
import torch.nn.functional as F

# 注意：从你的 seq2seq 模型文件导入
from models.model_history_transformer import RubikSeq2SeqTransformer
from tokenizer.tokenizer_rubik import RubikTokenizer

from utils import (
    cube_to_6x9,
    convert_state_to_tensor,  # 需兼容6x9输入 => (54,)
    MOVES_POOL,
    move_idx_to_str,
    PAD_TOKEN,
    MASK_OR_NOMOVE_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN
)

tokenizer = RubikTokenizer()

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

def build_src_tensor_from_cube(cube, history_len=8):
    """
    将推断时的初始状态，构造成与训练时相同的格式：
      - 形状: (1, history_len+1, 55)
      - 若 t=0，则代表「前 history_len 条状态」都无效，使用 PAD_TOKEN 填充
      - 最后一条 (index=-1) 填入当前实际状态的 54 维 + move 索引 (无真实 move 则用 -1)
    """
    # 1. 准备一个 (history_len+1, 55) 的张量，并全部用 PAD_TOKEN 填充
    seq_len = history_len + 1
    src_seq = torch.full((seq_len, 55), PAD_TOKEN, dtype=torch.long)

    # 2. 获取魔方当前状态的 54 维张量
    #    与训练时保持一致: (6x9) => (54,)
    s6x9 = cube_to_6x9(cube)
    state_54 = tokenizer.encode_state(s6x9)  # (54,)

    # 3. 在最后一行填入「当前状态 + move 索引MASK_OR_NOMOVE_TOKEN」
    src_seq[-1, :54] = state_54
    dummy_move = torch.tensor([MASK_OR_NOMOVE_TOKEN], dtype=torch.long)  # shape (1,)
    src_seq[-1, 54] = dummy_move

    # 4. 扩展一个 batch 维度 => (1, history_len+1, 55)
    return src_seq.unsqueeze(0)


@torch.no_grad()
def beam_search(model, cube, history_len=8, max_len=50, beam_size=3, device="cuda"):
    """
    使用 Beam Search（替代贪心）迭代预测动作序列，直到魔方复原或达到最大步数。

    与 iterative_greedy_decode_seq2seq 类似：
      - 每步根据最近 history_len 步的状态更新输入 src
      - 利用模型预测下一个动作（这里采用 Beam Search 而非单步 argmax）
      - 将预测动作应用到 cube 上，并更新状态
      - 如果预测到 EOS/PAD 或魔方复原，则提前结束

    参数:
      model: seq2seq 模型
      cube: 魔方对象（必须支持 cube_to_6x9、cube(move_str) 和 is_solved 等接口）
      history_len: 用于构造 src 的历史步数
      max_len: 最大预测步数
      beam_size: Beam Search 每步扩展的候选数量
      device: 设备

    返回:
      List[int]，预测出的 move token 序列
    """
    model.eval()

    # 初始化记录：steps 中存放 (state, move) 序列，初始状态不含动作
    steps = []
    init_state_6x9 = cube_to_6x9(cube)  # 用户需自行实现：将 cube 转为 6x9 表示
    steps.append((init_state_6x9, None))

    predicted_moves = []

    for t in range(max_len):
        # 根据最近 history_len 步构建 src，形状为 (1, history_len+1, feature_dim)
        src = build_src_tensor_from_steps(steps, history_len=history_len)
        src = src.to(device)

        # 使用 Beam Search 预测下一个动作：
        # 这里我们只展开一步（max_steps=1），因此返回的 tokens 序列应仅含一个 token
        next_tokens = _beam_search_step(model, src, beam_size=beam_size, max_steps=1, device=device)
        if not next_tokens:
            break
        next_token_id = next_tokens[0]

        # 若预测到 EOS 或 PAD，则结束推理
        if next_token_id == EOS_TOKEN or next_token_id == PAD_TOKEN:
            print("模型预测到EOS/PAD，推理结束.")
            break

        # 将 token 转为动作字符串，并应用到魔方上
        next_move_str = tokenizer.decode_move(next_token_id)
        cube(next_move_str)  # 应用动作
        predicted_moves.append(next_token_id)

        # 检查是否已复原
        if is_solved(cube):
            print(f"在第 {t + 1} 步成功复原!")
            break

        # 更新步骤记录：记录最新状态及刚执行的动作
        new_state_6x9 = cube_to_6x9(cube)
        steps.append((new_state_6x9, next_move_str))

    return predicted_moves


@torch.no_grad()
def _beam_search_step(model, src, beam_size=3, max_steps=1, device="cuda"):
    """
    基于给定 src 使用 Beam Search 解码，返回最佳候选的 tokens 序列。
    与原始 beam_search 实现一致，只是这里 max_steps 参数通常设置为1（即仅预测下一个动作）。
    """
    # 初始 decoder 输入为 [SOS_TOKEN]（不作为最终输出）
    initial_decoder = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    # 每个候选分支记录 decoder_input、累计对数概率以及已生成的 token 序列（不含 SOS）
    beam = [{
        "decoder_input": initial_decoder,
        "score": 0.0,
        "tokens": []
    }]
    finished_candidates = []

    for _ in range(max_steps):
        new_beam = []
        for candidate in beam:
            decoder_input = candidate["decoder_input"]  # 形状: (1, current_seq_len)
            # 前向传播，输出 logits，形状: (1, seq_len, num_moves)
            logits = model(src, decoder_input)
            # 取最后一步的 logits，形状: (1, num_moves)
            last_logits = logits[:, -1, :]
            # 计算 softmax 概率
            probs = F.softmax(last_logits, dim=-1)
            # 选取 topk 个候选动作
            topk_probs, topk_indices = torch.topk(probs, k=beam_size, dim=-1)
            topk_probs = topk_probs.squeeze(0)  # (beam_size,)
            topk_indices = topk_indices.squeeze(0)  # (beam_size,)

            # 扩展每个候选分支
            for prob, token in zip(topk_probs, topk_indices):
                token_id = token.item()
                new_score = candidate["score"] + torch.log(prob).item()
                new_tokens = candidate["tokens"] + [token_id]
                # 将新 token 添加到 decoder 输入中
                new_decoder_input = torch.cat(
                    [decoder_input, token.unsqueeze(0).unsqueeze(0)], dim=1
                )
                # 如果生成 EOS 或 PAD，则将该候选视为结束
                if token_id == EOS_TOKEN or token_id == PAD_TOKEN:
                    finished_candidates.append({
                        "decoder_input": new_decoder_input,
                        "score": new_score,
                        "tokens": new_tokens
                    })
                else:
                    new_beam.append({
                        "decoder_input": new_decoder_input,
                        "score": new_score,
                        "tokens": new_tokens
                    })
        # 如果没有扩展出新的候选，则退出循环
        if not new_beam:
            break
        # 选择得分最高的 beam_size 个候选继续扩展
        beam = sorted(new_beam, key=lambda x: x["score"], reverse=True)[:beam_size]

    # 如果存在已完成候选，则返回得分最高的；否则返回当前 beam 中得分最高的候选
    if finished_candidates:
        best_candidate = sorted(finished_candidates, key=lambda x: x["score"], reverse=True)[0]
    else:
        best_candidate = sorted(beam, key=lambda x: x["score"], reverse=True)[0]

    return best_candidate["tokens"]


@torch.no_grad()
def iterative_greedy_decode_seq2seq(model, cube, history_len=8, max_len=50, device = "cuda"):
    """
    一个示例：使用与训练时类似的“滑动窗口”逻辑做贪心解码推理。
    - 每步都更新 src，让 src 包含最近 history_len 步 (含当前步) 的 [状态 + 动作]
    - 仅预测“下一个动作”，然后更新魔方状态，再继续。
    """

    model.eval()

    # 用于记录已经执行的 (state, move) 序列，最初时没有 move，因此 move=None 或 -1
    # 这里 steps 的结构与训练时相同: [(s6x9_0, mv_0), (s6x9_1, mv_1), ...]
    steps = []
    # 先把初始状态放进 steps，move=None
    init_state_6x9 = cube_to_6x9(cube)  # 你要自行实现
    steps.append((init_state_6x9, None))

    predicted_moves = []
    for t in range(max_len):
        # 1. 根据 steps 的“最后 history_len 步”构建 (1, history_len+1, 55) 的 src
        src = build_src_tensor_from_steps(steps, history_len=history_len)  # 形如 (1, hist_len+1, 55)
        src = src.to(device)

        # 2. Decoder 端输入: [SOS_TOKEN], shape = (1,1)
        decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)

        # 3. 前向，拿到 logits => (1, 1, num_moves)
        logits = model(src, decoder_input)  # (batch=1, seq_len=1, num_moves=?)
        last_step_logits = logits[:, -1, :]  # => shape: (1, num_moves)

        # 4. 取 argmax 作为下一个动作
        next_token_id = torch.argmax(last_step_logits, dim=1).item()
        if next_token_id == EOS_TOKEN or next_token_id == PAD_TOKEN:
            print("模型预测到EOS/PAD，推理结束.")
            break

        # 5. 将该动作应用到魔方
        next_move_str = tokenizer.decode_move(next_token_id)
        cube(next_move_str)  # 执行动作
        predicted_moves.append(next_token_id)

        # 6. 检查是否复原
        if is_solved(cube):
            print(f"在第 {t+1} 步成功复原!")
            break

        # 7. 更新 steps: 新的状态 + 动作
        new_state_6x9 = cube_to_6x9(cube)
        steps.append((new_state_6x9, next_move_str))

    return predicted_moves

def build_src_tensor_from_steps(steps, history_len=8):
    # steps 的长度
    total_len = len(steps)

    # 先用 PAD_TOKEN 填充 (history_len+1, 55)
    src_seq = torch.full(
        (history_len + 1, 55),
        PAD_TOKEN,
        dtype=torch.long
    )

    # 找到本次要用的窗口: 从 max(0, total_len - (history_len+1)) 到 total_len
    start_idx = max(0, total_len - (history_len + 1))
    used_steps = steps[start_idx : total_len]

    # 用于将真实 steps 数据对齐到 src_seq 最右侧
    offset = (history_len + 1) - len(used_steps)

    for i, (s6x9_i, mv_i) in enumerate(used_steps):
        state_tensor = tokenizer.encode_state(s6x9_i)  # => (54,)
        src_seq[offset + i, :54] = state_tensor
        mv_idx = tokenizer.encode_move(mv_i)
        src_seq[offset + i, 54] = mv_idx

    # 扩展出 batch 维度 => (1, history_len+1, 55)
    return src_seq.unsqueeze(0)


@torch.no_grad()
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
    new_state_dict = OrderedDict()
    state_dict = torch.load(config.inference.model_path, map_location=device)
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(config.inference.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 随机打乱魔方
    scramble_steps = config.inference.scramble_steps
    cube, scramble_moves = random_scramble_cube(steps=scramble_steps)
    print(f"Scramble moves: {scramble_moves}")

    # 3. 构造 src
    # src_tensor = build_src_tensor_from_cube(cube,config.inference.max_len).to(device)
    # 4. 用 seq2seq 进行贪心解码，生成还原动作序列
    pred_tokens = iterative_greedy_decode_seq2seq(model, cube, max_len=config.inference.max_len,device=device)
    # pred_tokens = beam_search(model, src_tensor, max_len=config.inference.max_len)
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
            pred_moves.append(tokenizer.decode_move(t))
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
