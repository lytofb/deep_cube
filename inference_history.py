# inference_seq2seq.py

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

# 假设你原有的 Dataset/模型/Utils
from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer
from utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN  # 或者你自己定义好的几个特殊token

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

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

        # 根据 src 长度确定从哪一步开始评估
        # 例如：若 tgt 长度为 9，src 长度为 4，则 teacher forcing 长度为 8，
        # 而 src 中已经包含前 3 个 step，所以从索引 3 开始评估 (8-3=5 个 token)
        eval_start_index = src.size(1) - 1

        pred_tokens_eval = pred_tokens[:, eval_start_index:]
        target_output_eval = target_output[:, eval_start_index:]

        # 打印当前 batch 中前 5 个样本（整个 dataloader 中只打印前 5 个样本）
        batch_size = src.size(0)
        for i in range(batch_size):
            if printed_samples < 3:
                print(f"Sample {printed_samples}:")
                print("  src.o:                ", src[i].cpu().tolist())
                print("  src:                ", src[i].cpu()[:,-1])
                print("  tgt:                ", tgt[i].cpu().tolist())
                print("  pred_tokens_eval:   ", pred_tokens_eval[i].cpu().tolist())
                print("  target_output_eval: ", target_output_eval[i].cpu().tolist())
                printed_samples += 1
            else:
                break

        total_correct += (pred_tokens_eval == target_output_eval).sum().item()
        total_count += target_output_eval.numel()

    if total_count == 0:
        return 0.0
    return total_correct / total_count


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
    print(f"[Validation], Val_Acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
