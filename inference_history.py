# inference_seq2seq.py

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

# 假设你原有的 Dataset/模型/Utils
from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer
from tokenizer.tokenizer_rubik import RubikTokenizer
from utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_OR_NOMOVE_TOKEN, \
    convert_tensor_to_state_6x9, create_cube_from_6x9, cube_to_6x9, convert_state_to_tensor  # 或者你自己定义好的几个特殊token
import torch.nn.functional as F

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

tokenizer = RubikTokenizer()
def greedy_decode_seq2seq(
        model,
        src,  # (1, src_seq_len, 55)
        max_len=50,
        sos_token=20,
        eos_token=18
):
    """
    使用贪心解码对单条数据 (batch=1) 做推理。
    - src: 形状 (1, src_seq_len, 55)，对应一条输入序列
    - 返回: List[int]，预测出的 move token 序列（不含 SOS）
    """
    device = src.device
    model.eval()

    # 初始 decoder_input，含 [SOS]
    decoder_input = torch.tensor([[sos_token]], dtype=torch.long, device=device)  # shape=(1,1)

    predicted_tokens = []
    with torch.no_grad():
        for _ in range(max_len):
            # 前向: model(src, decoder_input)
            # logits => (1, 当前decoder长度, num_moves)
            logits = model(src, decoder_input)

            # 取最后一个时间步 => shape (1, num_moves)
            last_step_logits = logits[:, -1, :]
            next_token = torch.argmax(last_step_logits, dim=1)  # shape (1,)

            # 如果预测到EOS，就停止
            if next_token.item() == eos_token:
                break

            # 否则，把 next_token 拼到 decoder_input 末尾
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            predicted_tokens.append(next_token.item())

    return predicted_tokens


@torch.no_grad()
def evaluate_seq2seq_accuracy(model, dataloader, device):
    """
    对验证集做推断，并计算 token-level Accuracy：
      1. 同样用 teacher forcing，得到 logits
      2. 取 argmax
      3. 与 target 对比，统计正确率
    返回: float, 即正确率 (correct_tokens / total_tokens)
    """
    model.eval()

    total_correct = 0
    total_count = 0
    printed_samples = 0  # 用于记录已打印的样本数量

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 同训练方式 (Teacher forcing)
        decoder_input = tgt[:, :-1]
        target_output = tgt[:, 1:]  # 形状 (B, seq_len-1)

        logits = model(src, decoder_input)  # => (B, seq_len-1, num_moves)
        # 取 argmax => (B, seq_len-1)
        pred_tokens = logits.argmax(dim=-1)

        # 打印当前 batch 中前 5 个样本（整个 dataloader 中只打印前 5 个样本）
        batch_size = src.size(0)
        for i in range(batch_size):
            if printed_samples < 3:
                print(f"Sample {printed_samples}:")
                # print("  src.o:                ", src[i].cpu().tolist())
                print("  src:                ", src[i].cpu()[:,-1])
                print("  tgt:                ", tgt[i].cpu().tolist())
                print("  pred_tokens:   ", pred_tokens[i].cpu().tolist())
                print("  target_output: ", target_output[i].cpu().tolist())
                printed_samples += 1
            else:
                break

        total_correct += (pred_tokens == target_output).sum().item()
        total_count += target_output.numel()

    if total_count == 0:
        return 0.0
    return total_correct / total_count

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
        state_tensor = s6x9_i  # => (54,)
        src_seq[offset + i, :54] = state_tensor
        mv_idx = mv_i if mv_i is not None else MASK_OR_NOMOVE_TOKEN
        src_seq[offset + i, 54] = mv_idx

    # 扩展出 batch 维度 => (1, history_len+1, 55)
    return src_seq.unsqueeze(0)

@torch.no_grad()
def evaluate_seq2seq_accuracy_with_repetition_penalty(model, dataloader, device, history_len=8, max_len=50):
    """
    对验证集做推断，使用迭代贪心解码（free-run），并添加重复惩罚，避免连续生成相同的 token。
    只验证 dataloader 中 2 个样本。

    预测流程：
      - 以验证数据中提供的 src 的最后状态作为初始状态，构建 steps。
      - 每一步用 build_src_tensor_from_steps 构造输入张量，
        Decoder 输入固定为 [SOS_TOKEN]，预测下一个 token。
      - 在 logits 上将上一步预测的 token 对应的值设为 -inf，从而确保连续不会重复。
      - 当预测到 EOS 或 PAD 时停止解码。

    最后将预测得到的 token 序列（不包含 SOS）与 ground truth（同样跳过 SOS）比较，
    返回 token-level Accuracy。
    """
    model.eval()
    total_correct = 0
    total_count = 0
    sample_count = 0  # 已验证样本数

    # 仅验证 2 个样本
    for src, tgt in dataloader:
        batch_size = src.size(0)
        for i in range(batch_size):
            if sample_count >= 2:
                break

            # 取单个样本，假设 src 的最后一部分包含初始状态信息
            sample_src = src[i].unsqueeze(0).to(device)  # shape: (1, seq_len, feature_dim)
            sample_tgt = tgt[i].unsqueeze(0).to(device)  # shape: (1, tgt_seq_len)

            # 初始化 steps，使用 sample_src[0]（即该样本的状态）作为初始状态，move 为 None
            initial_state = sample_src[0, -1, :54]  # shape: (seq_len, feature_dim)
            steps = [(initial_state, None)]
            decoded_tokens = []
            decoder_tokens = [SOS_TOKEN]

            # 迭代贪心解码
            for t in range(max_len):
                # 构造最近 history_len 步的输入
                input_tensor = build_src_tensor_from_steps(steps,
                                                           history_len=history_len)  # shape: (1, history_len+1, feature_dim)
                input_tensor = input_tensor.to(device)

                # Decoder 输入固定为 [SOS_TOKEN]
                decoder_input = torch.tensor([decoder_tokens], dtype=torch.long, device=device)
                logits = model(input_tensor, decoder_input)  # (1, 1, num_moves)
                last_logits = logits[:, -1, :]  # shape: (1, num_moves)

                # 添加重复惩罚：若上一步已有预测，则将该 token 对应的 logit 置为 -∞
                if decoded_tokens:
                    prev_token = decoded_tokens[-1]
                    last_logits[0, prev_token] = -float('inf')

                # 选择下一个 token
                next_token_id = torch.argmax(last_logits, dim=1).item()
                if next_token_id == EOS_TOKEN:
                    break
                # if next_token_id == EOS_TOKEN or next_token_id == PAD_TOKEN:
                #     break
                decoded_tokens.append(next_token_id)
                decoder_tokens.append(next_token_id)

                # 更新 steps：这里仅作示例，状态保持不变。如果你有状态更新函数，
                new_state = update_state(steps[-1][0], next_token_id)
                steps.append((new_state, next_token_id))

            # 将 ground truth 转为列表，并跳过首个 SOS_TOKEN（假设 tgt[0] 为 SOS）
            ground_truth = sample_tgt[0].cpu().tolist()[1:]
            pred_tokens = decoded_tokens

            # 计算 token-level accuracy（以较短序列为准）
            min_len = min(len(pred_tokens), len(ground_truth))
            correct = sum(1 for j in range(min_len) if pred_tokens[j] == ground_truth[j])
            total_correct += correct
            total_count += min_len

            print(f"Sample {sample_count}:")
            print("  Predicted tokens: ", pred_tokens)
            print("  Ground truth:     ", ground_truth)
            sample_count += 1

        if sample_count >= 2:
            break

    if total_count == 0:
        return 0.0
    return total_correct / total_count

@torch.no_grad()
def _beam_search_step_with_repetition_penalty(model, src, beam, beam_size=3, max_steps=1, device="cuda"):
    """
    基于给定 src 使用 Beam Search 解码（仅扩展一步），并添加重复惩罚：
      - 如果某候选分支已经生成过 token，则将上一步生成的 token 对应的 logit 置为 -∞，
        从而确保该 token 不会在本次扩展中重复出现。
    返回最佳候选的 tokens 序列（不含 SOS）。
    """
    # 初始 decoder 输入为 [SOS_TOKEN]
    # initial_decoder = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    # beam = [{
    #     "decoder_input": initial_decoder,  # shape: (1, current_seq_len)
    #     "score": 0.0,
    #     "tokens": []  # 存储已生成的 token 序列（不含 SOS）
    # }]
    finished_candidates = []

    for _ in range(max_steps):
        new_beam = []
        for candidate in beam:
            decoder_input = candidate["decoder_input"]
            # 前向传播，得到 logits: (1, seq_len, num_moves)
            logits = model(src, decoder_input)
            # 取最后一步 logits，形状: (1, num_moves)
            last_logits = logits[:, -1, :]
            # 添加重复惩罚：若候选分支已有生成 token，则将上一次生成 token 对应的 logit 置为 -∞
            if candidate["tokens"]:
                last_token = candidate["tokens"][-1]
                last_logits[0, last_token] = -float('inf')
            # 计算 softmax 概率
            probs = F.softmax(last_logits, dim=-1)
            # 取 top-k 候选动作
            topk_probs, topk_indices = torch.topk(probs, k=beam_size, dim=-1)
            topk_probs = topk_probs.squeeze(0)      # (beam_size,)
            topk_indices = topk_indices.squeeze(0)  # (beam_size,)

            # 对每个候选扩展
            for prob, token in zip(topk_probs, topk_indices):
                token_id = token.item()
                new_score = candidate["score"] + torch.log(prob).item()
                new_tokens = candidate["tokens"] + [token_id]
                # 将新 token 添加到 decoder 输入中
                new_decoder_input = torch.cat(
                    [decoder_input, token.unsqueeze(0).unsqueeze(0)], dim=1
                )
                # 如果生成 EOS 或 PAD，则视为结束候选
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
        # 若没有新的候选，则退出
        if not new_beam:
            break
        # 选择得分最高的 beam_size 个候选
        beam = sorted(new_beam, key=lambda x: x["score"], reverse=True)[:beam_size]

    if finished_candidates:
        best_candidate = sorted(finished_candidates, key=lambda x: x["score"], reverse=True)[0]
    else:
        best_candidate = sorted(beam, key=lambda x: x["score"], reverse=True)[0]
    return best_candidate["tokens"]


@torch.no_grad()
def evaluate_seq2seq_accuracy_with_beam_search(model, dataloader, device, history_len=8, max_len=50, beam_size=16):
    """
    对验证集做推断，使用基于 Beam Search 的迭代解码（free-run），并在每一步添加重复惩罚（禁止连续生成相同 token）。
    只验证 dataloader 中 2 个样本。

    解码流程：
      - 从验证数据中取出 src 的最后一行（前54个元素）作为初始状态，
        构建 steps（状态 + 动作），初始时动作为 None。
      - 每一步调用 build_src_tensor_from_steps 构造输入张量，
        再利用 _beam_search_step_with_repetition_penalty 预测下一个 token。
      - 如果预测到 EOS 或 PAD，则结束解码。
      - 最后将预测出的 token 序列（不含 SOS）与 ground truth（跳过 SOS）比较，
        返回 token-level Accuracy。
    """
    model.eval()
    total_correct = 0
    total_count = 0
    sample_count = 0  # 已验证样本数

    # 仅验证 2 个样本
    for src, tgt in dataloader:
        batch_size = src.size(0)
        for i in range(batch_size):
            if sample_count >= 2:
                break

            # 取单个样本，src shape: (1, seq_len, 55)，tgt shape: (1, tgt_seq_len)
            sample_src = src[i].unsqueeze(0).to(device)
            sample_tgt = tgt[i].unsqueeze(0).to(device)

            # 使用 sample_src 的最后一行的前 54 个元素作为初始状态（要求与 build_src_tensor_from_steps 接口一致）
            initial_state = sample_src[0, -1, :54]  # shape: (54,)
            steps = [(initial_state, None)]
            decoded_tokens = []

            beam = [{
                "decoder_input": torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device),
                "score": 0.0,
                "tokens": []
            }]
            # 迭代解码
            for t in range(max_len):
                # 根据最近 history_len 步构造输入 tensor，形状: (1, history_len+1, 55)
                input_tensor = build_src_tensor_from_steps(steps, history_len=history_len).to(device)
                # 使用 Beam Search 进行一步扩展
                next_tokens = _beam_search_step_with_repetition_penalty(model, input_tensor, beam, beam_size=beam_size, max_steps=1, device=device)
                if not next_tokens:
                    break
                next_token_id = next_tokens[0]
                if next_token_id == EOS_TOKEN or next_token_id == PAD_TOKEN:
                    break
                decoded_tokens.append(next_token_id)
                # 更新 steps：此处示例中状态保持不变；实际使用时可调用状态更新函数
                new_state = update_state(steps[-1][0], next_token_id)
                steps.append((new_state, next_token_id))

            # 将 ground truth 转换为列表，并跳过首个 SOS_TOKEN（假设 tgt[0] 为 SOS）
            ground_truth = sample_tgt[0].cpu().tolist()[1:]
            pred_tokens = decoded_tokens

            # 计算 token-level accuracy（以较短序列长度为准）
            min_len = min(len(pred_tokens), len(ground_truth))
            correct = sum(1 for j in range(min_len) if pred_tokens[j] == ground_truth[j])
            total_correct += correct
            total_count += min_len

            print(f"Sample {sample_count}:")
            print("  Predicted tokens: ", pred_tokens)
            print("  Ground truth:     ", ground_truth)
            sample_count += 1

        if sample_count >= 2:
            break

    if total_count == 0:
        return 0.0
    return total_correct / total_count


def evaluate_seq2seq_accuracy_with_repetition_penalty_top_p(
        model, dataloader, device, history_len=8, max_len=50, p=0.9
):
    """
    对验证集做推断，使用迭代 Top-p (Nucleus) 采样解码，并添加重复惩罚，避免连续生成相同的 token。
    只验证 dataloader 中 2 个样本。

    Args:
        model: Seq2Seq Transformer 模型
        dataloader: 验证集的数据加载器
        device: 计算设备
        history_len: 构造输入时查看多少步历史状态
        max_len: 生成的最大长度
        p: top-p 采样的阈值 (nucleus size)，通常可选 0.8～0.95

    Returns:
        token-level accuracy (以较短序列为准做比较)
    """
    model.eval()
    total_correct = 0
    total_count = 0
    sample_count = 0  # 已验证样本数

    for src, tgt in dataloader:
        batch_size = src.size(0)
        for i in range(batch_size):
            if sample_count >= 5:
                break

            # 取单个样本
            sample_src = src[i].unsqueeze(0).to(device)  # shape: (1, seq_len, feature_dim)
            sample_tgt = tgt[i].unsqueeze(0).to(device)  # shape: (1, tgt_seq_len)

            # 初始化 steps (示例：魔方状态 + token)
            initial_state = sample_src[0, -1, :54]  # 取最后的那步状态
            steps = [(initial_state, None)]

            # decoder_input 首次只包含 [SOS_TOKEN]
            decoder_tokens = [SOS_TOKEN]
            decoded_tokens = []

            for t in range(max_len):
                # 1) 构造最近 history_len 步的输入 (encoder 部分)
                input_tensor = build_src_tensor_from_steps(steps, history_len=history_len)
                input_tensor = input_tensor.to(device)

                # 2) 将当前已解码序列（包括 SOS_TOKEN）打包给 Decoder
                #    这里 decoder_tokens 是一个 list[int]，需要变成 (1, seq_len_so_far)
                decoder_input = torch.tensor([decoder_tokens], dtype=torch.long, device=device)

                # 3) 前向计算：输出维度 (B=1, seq_len_so_far, num_moves)
                logits = model(input_tensor, decoder_input)
                last_logits = logits[:, -1, :]  # shape: (1, num_moves)  取当前时刻的输出

                # 4) 重复惩罚(演示：若上一时刻生成过 token，则将其 logit 设为 -∞)
                if len(decoded_tokens) > 0:
                    prev_token = decoded_tokens[-1]
                    last_logits[0, prev_token] = -float('inf')

                # 5) 做 softmax 转为概率分布，然后 top-p 采样
                probs = F.softmax(last_logits, dim=-1).squeeze(0)  # shape: (num_moves,)
                next_token_id = top_p_sampling(probs, p=p)

                # 6) 若预测到 EOS_TOKEN 则停止
                if next_token_id == EOS_TOKEN:
                    break

                # 记录解码结果
                decoded_tokens.append(next_token_id)
                # 同时扩展 decoder_tokens，使下一轮解码能看到自己已生成的所有token
                decoder_tokens.append(next_token_id)

                # 7) 状态更新 (若你有魔方状态更新函数)
                new_state = update_state(steps[-1][0], next_token_id)
                steps.append((new_state, next_token_id))

            # -- 生成结束，计算和真实序列的 token-level accuracy
            # 跳过真实序列中开头的 SOS_TOKEN
            ground_truth = sample_tgt[0].cpu().tolist()[1:]
            pred_tokens = decoded_tokens

            min_len = min(len(pred_tokens), len(ground_truth))
            correct = sum(1 for j in range(min_len) if pred_tokens[j] == ground_truth[j])
            total_correct += correct
            total_count += min_len

            print(f"Sample {sample_count}:")
            print("  Predicted tokens: ", pred_tokens)
            print("  Ground truth:     ", ground_truth)
            sample_count += 1

        if sample_count >= 5:
            break

    if total_count == 0:
        return 0.0
    return total_correct / total_count


def top_p_sampling(probabilities: torch.Tensor, p=0.9):
    """
    在给定的概率分布 (1D Tensor) 上进行 Top-p (Nucleus) 采样。
    返回采样得到的 token 索引。

    Args:
        probabilities: shape (vocab_size,), 已经过 softmax
        p: 阈值, 累积概率到 p 则停止

    Returns:
        int: 采样得到的 token 索引
    """
    # 1. 对概率从大到小排序
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # 2. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # 3. 找到第一个使 cumulative_probs >= p 的位置
    cutoff_idx = torch.searchsorted(cumulative_probs, p)

    # 4. 截断到这个cutoff_idx（也可+1，确保至少包含该位置token）
    #    例如 cumulative_probs=[0.3,0.5,0.6,0.9, ...]，若p=0.7则 cutoff_idx=2
    #    只取 sorted_indices[:3] => 0,1,2
    truncated_indices = sorted_indices[:cutoff_idx+1]
    truncated_probs = sorted_probs[:cutoff_idx+1]

    # 5. 在截断后的概率分布中做归一化，并进行一次随机采样
    truncated_probs = truncated_probs / truncated_probs.sum()

    # 6. 从 truncated_probs 中随机采样
    next_token_id = torch.multinomial(truncated_probs, 1)

    # 7. 获取在原始 vocab 中对应的 token 索引
    next_token_id = truncated_indices[next_token_id]
    return next_token_id.item()


def update_state(old_state_tensor, next_token_id):
    """
    根据旧状态(54 维张量)和下一个动作 token_id，
    使用 PyCuber 执行 move，并返回新状态(54 维张量)。

    Args:
        old_state_tensor: shape (54,) 的 LongTensor，表示魔方当前的颜色状态
        next_token_id: int，下一个 move 的 token 索引

    Returns:
        new_state_tensor: shape (54,) 的 LongTensor，表示执行 move 后的新状态
    """
    # 1) 从旧的 54 维张量 -> 6x9 颜色布局
    old_state_6x9 = convert_tensor_to_state_6x9(old_state_tensor)

    # 2) 构造 pycuber.Cube
    cube = create_cube_from_6x9(old_state_6x9)

    # 3) 根据 token_id 找到动作字符串，然后执行 move
    move_str = tokenizer.decode_move(next_token_id)
    cube(move_str)

    # 4) 读出新的 6x9 颜色布局
    new_state_6x9 = cube_to_6x9(cube)

    # 5) 调用你的 convert_state_to_tensor，把 6x9 布局转回 54 维张量
    new_state_tensor = convert_state_to_tensor(new_state_6x9)

    return new_state_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载验证集
    # 假设你把验证集文件放在 rubik_val_shards 目录
    val_dataset = RubikDataset(data_dir=config.data.val_dir,
                               history_len=config.data.max_history_len,
                               max_files=None)
    # 如果你想批量处理，也可以做 DataLoader，但这里为了逐条解码方便，直接用 dataset[i] 就行

    # 2. 加载训练好的 seq2seq 模型
    model = RubikSeq2SeqTransformer(
        input_dim=config.model.input_dim,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        nhead=config.model.nhead,
        num_moves=config.model.num_moves,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.train.prefetch_factor
    )
    new_state_dict = OrderedDict()
    state_dict = torch.load(config.inference.model_path, map_location=device)
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    val_acc = evaluate_seq2seq_accuracy(model, val_loader, device)
    evaluate_seq2seq_accuracy_with_repetition_penalty(model, val_loader,device)
    # evaluate_seq2seq_accuracy_with_beam_search(model, val_loader,device)
    # evaluate_seq2seq_accuracy_with_repetition_penalty_top_p(model,val_loader,device)
    print(f"[Validation], Val_Acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
