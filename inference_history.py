# inference_seq2seq.py

import torch
from torch.utils.data import DataLoader

# 假设你原有的 Dataset/模型/Utils
from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer
from utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN  # 或者你自己定义好的几个特殊token


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


def evaluate_seq2seq_accuracy(
        model,
        dataset,
        device,
        max_len=50,
        sos_token=SOS_TOKEN,
        eos_token=EOS_TOKEN
):
    """
    在给定的 dataset 上循环做贪心解码，并计算 token-level accuracy。
    - dataset: 里每个样本是 (src_seq, tgt_seq)，其中:
       src_seq.shape=(history_len+1, 55)
       tgt_seq: [SOS, move1, move2, ..., moveN, EOS]
    - 返回平均的 token-level accuracy
    """

    model.eval()

    # 统计：预测正确的 token 数量 / 总 token 数量
    total_correct = 0
    total_tokens = 0

    for idx in range(len(dataset)):
        src_seq, tgt_seq = dataset[idx]
        # src_seq => (history_len+1, 55)
        # tgt_seq => 1D, [SOS, move1, move2, ..., moveN, EOS]

        # 构造 batch=1 的输入 => shape (1, history_len+1, 55)
        src_seq = src_seq.unsqueeze(0).to(device)  # (1, seq_len, 55)

        # 用贪心解码得到预测序列(不含 SOS, 不含 EOS)
        pred_tokens = greedy_decode_seq2seq(
            model,
            src_seq,
            max_len=max_len,
            sos_token=sos_token,
            eos_token=eos_token
        )

        # 取 ground truth(不含 SOS/EOS) => tgt_seq[1:-1]
        # 这里假设: tgt_seq[0] = SOS, tgt_seq[-1] = EOS
        gt_tokens = tgt_seq[1:-1].tolist()  # 把 tensor转为 list

        # 对齐长度后，计算 token-level 准确率
        # 先找较短长度:
        min_len = min(len(pred_tokens), len(gt_tokens))
        correct_count = sum(
            p == g for (p, g) in zip(pred_tokens[:min_len], gt_tokens[:min_len])
        )
        total_correct += correct_count
        total_tokens += len(gt_tokens)  # 或者也可以只算 min_len

    # 避免除0
    if total_tokens == 0:
        return 0.0
    return total_correct / total_tokens


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载验证集
    # 假设你把验证集文件放在 rubik_val_shards 目录
    val_dataset = RubikDataset(data_dir='rubik_val_shards', max_files=None)
    # 如果你想批量处理，也可以做 DataLoader，但这里为了逐条解码方便，直接用 dataset[i] 就行

    # 2. 加载训练好的 seq2seq 模型
    model = RubikSeq2SeqTransformer(
        num_layers=4,
        d_model=2048
    )
    model.load_state_dict(torch.load("rubik_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 3. 在验证集上计算 token-level Accuracy
    acc = evaluate_seq2seq_accuracy(
        model,
        val_dataset,
        device,
        max_len=50,  # 最大解码长度
        sos_token=20,
        eos_token=18
    )

    print(f"Validation Token-level Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
