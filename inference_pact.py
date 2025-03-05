# inference_pact.py

import torch
import random
import pycuber as pc

# 从你定义的 GPT 模型文件导入
from models.model_pact_transformer import RubikGPT
from tokenizer.tokenizer_rubik import RubikTokenizer

tokenizer = RubikTokenizer()

from utils import (
    convert_state_to_tensor,
    MOVES_POOL,
    move_idx_to_str,
    PAD_TOKEN,
    EOS_TOKEN
)

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")


import torch
import torch.nn.functional as F
import pycuber as pc

@torch.no_grad()
def beam_search_pact(
    model,
    cube,
    beam_size=3,
    max_steps=50
):
    """
    使用 Beam Search 在 GPT/Pact 模型上进行解码:
      - 每个 "时间步" 包含 [state, action] 2 个 token,
      - 我们将最后的 action token logits 用 softmax 得到概率分布, 选 topK (即 beam_size),
      - 对每个候选动作创建新分支(克隆魔方并执行该动作),
      - 重复直到魔方复原或达到 max_steps.

    返回: (best_moves: List[str]) => 最优分支的动作序列
    """
    device = next(model.parameters()).device
    model.eval()

    # 每个候选分支的数据结构
    # 包含：
    #   - seq_list: List[Tensor], 表示连续 (T, 55)  (每个元素是 (55,) 的 state+action)
    #   - moves:  已执行动作的字符串列表
    #   - cube:   当前魔方对象 (pycuber)
    #   - log_prob: 当前分支的对数概率

    class BeamCandidate:
        def __init__(self, seq_list, moves, cube, log_prob):
            self.seq_list = seq_list
            self.moves = moves
            self.cube = cube
            self.log_prob = log_prob

    # 1) 初始分支: 仅含一个 (cube) + dummy action
    init_tensor = build_state_action_tensor(cube, action_idx=PAD_TOKEN)
    init_seq_list = [init_tensor]
    init_moves = []
    init_log_prob = 0.0  # 对数概率(累乘=累加)

    init_candidate = BeamCandidate(
        seq_list=init_seq_list,
        moves=init_moves,
        cube=cube,  # 注意：这是传引用, 之后会克隆
        log_prob=init_log_prob
    )

    # 当前 beam 的候选
    beam = [init_candidate]

    for step in range(max_steps):
        # 若 beam 中任何分支都可以复原魔方, 就可以提前退出
        for cand in beam:
            if is_solved(cand.cube):
                # 找到一个已解分支, 直接返回它的 moves
                return cand.moves

        # 2) 展开下一步:
        #    对 beam 中每个分支, forward -> 取最后 action token logits -> 选 topK 扩展
        all_next_candidates = []

        for cand in beam:
            # 先把 seq_list 堆叠 => (T,55), 加batch=1 => (1,T,55)
            seq_tensor = torch.stack(cand.seq_list, dim=0).unsqueeze(0).to(device)
            T = seq_tensor.size(1)

            # forward => (1, 2T, vocab_size)
            logits = model(seq_tensor)

            # 取最后一个 action token => index= 2*T - 1
            last_action_logits = logits[:, 2*T - 1, :]  # (1, vocab_size)
            # 对它做 log_softmax => 得到各动作的对数概率 => shape(1, vocab_size)
            log_probs = F.log_softmax(last_action_logits, dim=-1).squeeze(0) # => (vocab_size,)

            # 选 topK (beam_size)
            topk = torch.topk(log_probs, k=beam_size, dim=-1)  # (values, indices)
            topk_logp = topk.values
            topk_idx = topk.indices

            # 针对每个候选动作, 生成新分支
            for i in range(beam_size):
                action_idx = topk_idx[i].item()
                action_logp = topk_logp[i].item()

                # 如果是 PAD_TOKEN 或 EOS_TOKEN, 就视为结束分支
                if action_idx == PAD_TOKEN or action_idx == EOS_TOKEN:
                    # 这里可以选择直接不扩展(分支终止)
                    # 或者创建一个cand副本并保留.
                    # 简单起见, 我们不扩展(即不再产生后续动作).
                    continue

                # 新分支 => 深拷贝cube, 执行该动作
                new_cube = clone_cube(cand.cube)
                mv_str = tokenizer.decode_move(action_idx)
                new_cube(mv_str)

                # 构建下一个 (55,) 的 state+action (action先用PAD占位)
                #   其实, 这里**如果**你想在 GPT 输入中记录 "上一步动作"=action_idx
                #   则 next 时, action_idx 不能是 PAD, 而是**上步刚生成的**action
                #   这要看你在训练时对 "state+action" 的约定。
                #   通常 PACT 里 "src[..., 54]" = 上一步动作. 这里我们**确实**要用 action_idx
                #   以表明 "上一步动作" = action_idx
                #   但**下一步** state+action 的**第 54 维**(=action)先留 PAD ?
                #   取决于你实现.
                #   —— 以下演示: "本步" => state(t), action(t), "下一步" => state(t+1), action(t+1)=pad
                #   其实和 greedy_decode_pact 一样,
                #   每次 forward 的 "最后action" 用来做 decode,
                #   "下一步" 还要 append (state+PAD).
                #
                #   不过这里先演示**简化**:
                #   当我们选择了 action_idx, 这个**其实**是 "本步" action,
                #   计算下个输入 token 还需要 "魔方新状态+ dummy action".
                #   => 参考 greedy_decode_pact() 里的做法.

                # 先**更新**cand序列: cand的最后(=当前)那一条 (55,) 里 action 位(54) 用 action_idx
                # 但实际上, cand.seq_list里**已经**含有 (state, action=PAD) ??
                #   => 其实 GPT 要 "2T" tokens: (state(t), action(t)),
                #      cand.seq_list[-1] 里 action = PAD,
                #      现在我们要把它改成 action=action_idx => "本步"就完成了
                new_seq_list = list(cand.seq_list)  # 浅拷贝list
                current_sa = new_seq_list[-1].clone()  # (55,)
                current_sa[54] = action_idx  # 把最后一维改成action_idx
                new_seq_list[-1] = current_sa

                # 再生成下一步 token => (state(t+1), action=PAD)
                new_sa_next = build_state_action_tensor(new_cube, action_idx=PAD_TOKEN)
                new_seq_list.append(new_sa_next)

                # 构建新的 moves 列表
                new_moves = cand.moves + [mv_str]

                # 新分支的对数概率
                new_log_prob = cand.log_prob + action_logp

                new_cand = BeamCandidate(
                    seq_list=new_seq_list,
                    moves=new_moves,
                    cube=new_cube,
                    log_prob=new_log_prob
                )
                all_next_candidates.append(new_cand)

        # 若本轮所有分支都没扩展(例如都输出EOS/PAD)，说明无法继续 => 直接退出
        if len(all_next_candidates) == 0:
            break

        # 3) 从 all_next_candidates 中选出 log_prob 最大的 beam_size 个
        #    就是 beam search 的精髓
        all_next_candidates.sort(key=lambda c: c.log_prob, reverse=True)
        beam = all_next_candidates[:beam_size]

    # 如果到这里还没复原, 就返回 beam[0] 的动作序列(最高 log_prob 分支)
    best_cand = beam[0]
    return best_cand.moves


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


def build_state_action_tensor(cube, action_idx=PAD_TOKEN):
    """
    将魔方当前状态(54维) + 给定的action_idx(1维) 拼成 (55,) 的 Tensor。
    这里默认 action_idx=PAD_TOKEN 作为“占位”。
    """
    # 1) 先获取 6x9 的颜色表示
    s6x9 = []
    for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
        face_obj = cube.get_face(face_name)
        row_data = []
        for r in range(3):
            for c in range(3):
                color_char = str(face_obj[r][c]).strip('[]').lower()
                row_data.append(color_char)
        s6x9.append(row_data)

    # 2) 转成 (54,) 的向量（int 索引等）
    state_54 = tokenizer.encode_state(s6x9)  # => shape (54,)

    # 3) 拼上 action => (55,)
    #    注意 GPT 里第 54 维是“上一步动作”
    action_tensor = torch.tensor([action_idx], dtype=torch.long)
    combined_55 = torch.cat([state_54, action_tensor], dim=0)
    return combined_55  # => (55,)


@torch.no_grad()
def greedy_decode_pact(model, cube, max_steps=50):
    """
    PACT / GPT 自回归贪心解码示例。
    - 起始输入：只含有当前魔方状态 + dummy action
    - 每一步：模型输出 => 取最后 action token => 得到动作 => 应用到魔方 => 得到新状态，append 到序列
    - 直到魔方复原或到达 max_steps
    返回：预测到的动作序列 (str 列表)
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) 构造一个 list，用来存储 [state+action] 的序列。
    #    初始只有一条 (当前魔方状态 + dummy action)
    seq_list = []
    init_tensor = build_state_action_tensor(cube, action_idx=PAD_TOKEN)
    seq_list.append(init_tensor)  # 此时 T=1

    predicted_actions = []

    for step in range(max_steps):
        # 2) 把 seq_list 堆叠成 (T, 55)，再加上 batch=1 => (1, T, 55)
        src_tensor = torch.stack(seq_list, dim=0).unsqueeze(0).to(device)  # => (1, T, 55)

        # 3) 前向 => (1, 2T, vocab_size)
        logits = model(src_tensor)
        seq_len = src_tensor.size(1)  # T
        # 取最后一个 action token 的 logits，索引= 2*T-1
        last_action_logits = logits[:, 2*seq_len - 1, :]  # => (1, vocab_size)

        # 4) 贪心取 argmax
        next_action_idx = torch.argmax(last_action_logits, dim=-1).item()

        # 如果输出的是 PAD_TOKEN / EOS_TOKEN，就停止
        if next_action_idx == PAD_TOKEN or next_action_idx == EOS_TOKEN:
            break

        # 5) 将动作转成可执行字符串，并应用到魔方
        next_action_str = tokenizer.decode_move(next_action_idx)
        predicted_actions.append(next_action_str)
        cube(next_action_str)

        # 如果魔方已复原，就直接退出
        if is_solved(cube):
            break

        # 6) 获取新的魔方状态，再加一个 dummy action，append 到 seq_list
        new_tensor = build_state_action_tensor(cube, action_idx=next_action_idx)
        seq_list.append(new_tensor)  # T += 1

    return predicted_actions


def clone_cube(cube):
    """
    PyCuber 并没有内置完整的 copy()，但我们可以先把魔方状态转成字符串，再新建一个 cube。
    也可以用更高级的复制方法。
    """
    import copy
    # 如果你发现 pycuber.Cube 在高版本可用 copy.deepcopy，则可以直接用:
    # return copy.deepcopy(cube)

    # 否则，就把魔方状态转为公式字符串，然后再新建一个 Cube 执行该公式
    new_cube = pc.Cube()
    # 遍历原 cube 的每个面的贴纸，赋值给 new_cube
    # 也可把原 cube 还原操作序列记录，再在 new_cube 上 apply
    # 这里给一个最简思路： 由 cube 转换为 facelets，再给 new_cube 逐一赋值

    for face_name in ['U', 'L', 'F', 'R', 'B', 'D']:
        face_obj = cube.get_face(face_name)
        new_face_obj = new_cube.get_face(face_name)
        for r in range(3):
            for c in range(3):
                new_face_obj[r][c] = face_obj[r][c]
    return new_cube


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载训练好的 PACT GPT 模型
    model = RubikGPT(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        max_seq_len=config.model.max_seq_len,
        ff_dim=config.model.d_model * 4,
        dropout=config.model.dropout,
        vocab_size=config.model.num_moves
    )
    model.load_state_dict(torch.load(config.inference.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2) 随机打乱魔方
    scramble_steps = config.inference.scramble_steps
    cube, scramble_moves = random_scramble_cube(steps=scramble_steps)
    print(f"Scramble moves: {scramble_moves}")

    # 3) 用自回归贪心解码
    max_len = config.inference.max_len
    pred_moves = greedy_decode_pact(model, cube, max_steps=max_len)
    # pred_moves = beam_search_pact(model, cube, max_steps=max_len)
    print(f"Predicted moves: {pred_moves}")

    # 4) 结果检查
    if is_solved(cube):
        print(f"成功在 {len(pred_moves)} 步内复原魔方!")
    else:
        print(f"未能在 {len(pred_moves)} 步内复原魔方，可以尝试调整模型或增加 max_steps")


if __name__ == "__main__":
    main()
