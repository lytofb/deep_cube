# inference_seq2seq.py
import pickle

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

# 假设你原有的 Dataset/模型/Utils
from dataset_rubik import RubikDataset, collate_fn
from models.model_encoder_only import RubikEncoderOnly
from tokenizer.tokenizer_rubik import RubikTokenizer
from utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_OR_NOMOVE_TOKEN, \
    convert_tensor_to_state_6x9, create_cube_from_6x9, cube_to_6x9, convert_state_to_tensor  # 或者你自己定义好的几个特殊token
from cube_rotate import move_cube
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
        target_output = tgt[:, 1]  # 形状 (B, seq_len-1)

        logits = model(src)  # => (B, seq_len-1, num_moves)
        # 取 argmax => (B, seq_len-1)
        pred_tokens = logits.argmax(dim=-1)

        # 打印当前 batch 中前 5 个样本（整个 dataloader 中只打印前 5 个样本）
        batch_size = src.size(0)
        for i in range(batch_size):
            if printed_samples < 3:
                print(f"Sample {printed_samples}:")
                # print("  src.o:                ", src[i].cpu().tolist())
                print("  src:                ", src[i].cpu()[:, -1])
                print("  tgt:                ", tgt[i].cpu().tolist())
                print("  pred_tokens:   ", pred_tokens[i].cpu().tolist())
                print("  target_output: ", target_output[i].cpu().tolist())
                printed_samples += 1
            else:
                break

        mask = target_output != PAD_TOKEN
        correct = (pred_tokens == target_output) & mask
        total_correct += correct.sum().item()
        total_count += mask.sum().item()

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
    used_steps = steps[start_idx: total_len]

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
            if sample_count >= 5:
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
                logits = model(input_tensor) # shape: (1, num_moves)
                last_logits = logits.clone()  # shape: (1, num_moves)  取当前时刻的输出

                # 添加重复惩罚：若上一步已有预测，则将该 token 对应的 logit 置为 -∞
                # if decoded_tokens:
                #     prev_token = decoded_tokens[-1]
                #     last_logits[0, prev_token] = -float('inf')

                # 1) 将 last_logits 转为概率分布
                temperature = 1.2  # 例如 1.2，>1 会使分布更平滑；<1 会使分布更尖锐

                # 5) 做 softmax 转为概率分布，然后 top-p 采样
                #    在 softmax 之前先用 “/ temperature” 调整 logits
                probs = F.softmax(last_logits / temperature, dim=-1).squeeze(0)  # shape: (num_moves,)

                # 打印当前步骤、decoder_tokens 和 steps 信息
                print(f"[Step {t}] Steps:")
                # 如果 steps 是一个列表，每个元素可能包含多个信息，这里逐个打印
                for idx, step in enumerate(steps):
                    print(f"  Step {idx}: {step}")

                print(f"[Step {t}] Decoder tokens: {decoder_tokens}")

                # 2) 打印当前的 state (steps[-1][0]) 以及概率向量
                print(f"[Step {t}] Current State:")
                print(steps[-1][0].cpu().numpy().tolist())  # 或者只打印部分

                print(f"[Step {t}] Logits Probabilities:")
                # 获取 numpy 数组，并格式化每个概率为科学计数法，保留两位小数
                probs_np = probs.cpu().numpy()
                formatted_probs = [f"{p:.2e}" for p in probs_np]
                print(formatted_probs)

                # 选择下一个 token
                next_token_id = torch.argmax(last_logits, dim=1).item()
                # if next_token_id == EOS_TOKEN:
                #     break
                if next_token_id == EOS_TOKEN or next_token_id == PAD_TOKEN:
                    break
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

        if sample_count >= 5:
            break

    if total_count == 0:
        return 0.0
    return total_correct / total_count

@torch.no_grad()
def evaluate_free_run_success_rate(
    model,
    dataloader,
    device,
    history_len: int = 8,
    max_len: int = 50,
    sample_count: int = 5,
):
    """
    对 dataloader 中的样本做 free-run 解码，总共取 sample_count 个样本，
    当预测序列与 ground truth 在遇到第一个值为 19 的位置之前完全一致时，视为一次成功。

    返回:
      success_rate (float): 成功样本数 / sample_count
    """
    model.eval()
    successes = 0
    seen = 0
    diff_start_counts = {}
    first_mismatch_records = []

    for src_batch, tgt_batch in dataloader:
        batch_size = src_batch.size(0)
        for i in range(batch_size):
            if seen >= sample_count:
                break

            # —— 准备单样本 —— #
            sample_src = src_batch[i].unsqueeze(0).to(device)   # (1, seq_len, feat_dim)
            sample_tgt = tgt_batch[i].cpu().tolist()            # 包含 SOS_TOKEN

            # 跳过第一个 SOS，得到 ground truth 序列
            gt = sample_tgt[1:]

            # 找到 ground truth 中第一次出现 19 的位置（不包含该值本身）
            try:
                cutoff = gt.index(19)
            except ValueError:
                cutoff = len(gt)

            # —— free-run 解码 —— #
            # 初始化 steps 列表：[(state_tensor, move_id), ...]
            # 假设 state 保存在 src 序列的最后一项的前 54 维中
            init_state = sample_src[0, -1, :54]
            steps = [(init_state, None)]
            decoded = []

            for t in range(max_len):
                # 构建最近 history_len 步的模型输入
                inp = build_src_tensor_from_steps(steps, history_len=history_len)  # (1, history_len+1, feat_dim)
                inp = inp.to(device)

                # 直接调用模型，得到 (1, num_moves) 的 logits
                logits = model(inp).squeeze(0)  # (num_moves,)
                # 贪心选最大 logit，对应的 token
                tok = int(logits.argmax().item())
                # 强制让第一个tok设置为gt
                if t == 0 and tok != gt[0]:
                    first_mismatch_records.append({
                        "gt": gt,
                        "decoded": decoded,
                        "predict": tok,
                        "init_state": init_state
                    })
                    tok = gt[0]
                if tok in (EOS_TOKEN, PAD_TOKEN):
                    break

                decoded.append(tok)
                # 更新状态——假如你已有状态更新函数
                new_state = update_state(steps[-1][0], tok)
                steps.append((new_state, tok))

            # —— 判断是否成功 —— #
            # 只有当 decoded 在 [0:cutoff] 完全与 gt 在 [0:cutoff] 一致，才算成功
            # 判断成功与否，并统计首次出错位置
            if decoded[:cutoff] == gt[:cutoff]:
                successes += 1
            else:
                for j in range(cutoff):
                    if j >= len(decoded) or decoded[j] != gt[j]:
                        diff_start_counts[j] = diff_start_counts.get(j, 0) + 1
                        break

            seen += 1

        if seen >= sample_count:
            break

    success_rate = successes / sample_count if sample_count > 0 else 0.0
    return success_rate, diff_start_counts, first_mismatch_records

@torch.no_grad()
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
            if sample_count >= 2:
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

                # 3) 前向计算：输出维度 (B=1, seq_len_so_far, num_moves)
                logits = model(input_tensor) # shape: (1, num_moves)
                last_logits = logits.clone()  # shape: (1, num_moves)  取当前时刻的输出

                # 4) 重复惩罚(演示：若上一时刻生成过 token，则将其 logit 设为 -∞)
                # if len(decoded_tokens) > 0:
                #     prev_token = decoded_tokens[-1]
                #     last_logits[0, prev_token] = -float('inf')

                temperature = 1.5  # 例如 1.2，>1 会使分布更平滑；<1 会使分布更尖锐

                # 5) 做 softmax 转为概率分布，然后 top-p 采样
                #    在 softmax 之前先用 “/ temperature” 调整 logits
                probs = F.softmax(last_logits / temperature, dim=-1).squeeze(0)  # shape: (num_moves,)
                # 2) 打印当前的 state (steps[-1][0]) 以及概率向量
                print(f"[Step {t}] Current State:")
                print(steps[-1][0].cpu().numpy().tolist())  # 或者只打印部分

                print(f"[Step {t}] Logits Probabilities:")
                # 获取 numpy 数组，并格式化每个概率为科学计数法，保留两位小数
                probs_np = probs.cpu().numpy()
                formatted_probs = [f"{p:.2e}" for p in probs_np]
                print(formatted_probs)

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

        if sample_count >= 2:
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
    truncated_indices = sorted_indices[:cutoff_idx + 1]
    truncated_probs = sorted_probs[:cutoff_idx + 1]

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

    # 3) 根据 token_id 找到动作字符串，然后执行 move
    move_str = tokenizer.decode_move(next_token_id)
    new_state_6x9 = move_cube(old_state_6x9,move_str)

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
    model = RubikEncoderOnly(
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
        shuffle=True,
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
    # print("==============evaluate_seq2seq_accuracy_with_repetition_penalty==============")
    # evaluate_seq2seq_accuracy_with_repetition_penalty(model, val_loader, device)
    print("==============evaluate_free_run_success_rate==============")
    success_rate, diff_start_counts, first_mismatch_records = evaluate_free_run_success_rate(model, val_loader, device, sample_count=1000)
    print("========")
    print(success_rate)
    print(diff_start_counts)
    with open("first_mismatch_records.pkl", "wb") as f:
        pickle.dump(first_mismatch_records, f)
    # print("==============evaluate_seq2seq_accuracy_with_repetition_penalty_top_p==============")
    # evaluate_seq2seq_accuracy_with_repetition_penalty_top_p(model, val_loader, device, p=0.9)
    print(f"[Validation], Val_Acc={val_acc:.4f}")


if __name__ == "__main__":
    # {'U': 0, "U'": 1, 'U2': 2, 'D': 3, "D'": 4, 'D2': 5,
    #  'L': 6, "L'": 7, 'L2': 8, 'R': 9, "R'": 10, 'R2': 11,
    #  'F': 12,"F'": 13, 'F2': 14, 'B': 15, "B'": 16, 'B2': 17}
    main()
