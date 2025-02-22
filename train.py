# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR


def train_one_epoch_seq2seq(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc="Training"):
        # src: (B, src_seq_len, 55)，tgt: (B, tgt_seq_len)
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Teacher Forcing：外部切片
        decoder_input = tgt[:, :-1]  # (B, tgt_seq_len - 1)
        target_output = tgt[:, 1:]   # (B, tgt_seq_len - 1)

        # 前向传播
        logits = model(src, decoder_input)  # => (B, tgt_seq_len-1, num_moves)

        # 展平计算损失
        B, seq_len, num_moves = logits.shape
        logits = logits.reshape(-1, num_moves)       # => (B*(seq_len-1), num_moves)
        target_output = target_output.reshape(-1)    # => (B*(seq_len-1))

        loss = criterion(logits, target_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src.size(0)

    return total_loss / len(dataloader.dataset)


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

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 同训练方式 (Teacher forcing)
        decoder_input = tgt[:, :-1]
        target_output = tgt[:, 1:]  # 形状 (B, seq_len-1)

        logits = model(src, decoder_input)  # => (B, seq_len-1, num_moves)
        # 取 argmax => (B, seq_len-1)
        pred_tokens = logits.argmax(dim=-1)

        # 对齐 target_output => (B, seq_len-1)
        # 统计预测正确的数量
        total_correct += (pred_tokens == target_output).sum().item()
        total_count += target_output.numel()

    if total_count == 0:
        return 0.0
    return total_correct / total_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir='rubik_shards', max_files=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # ====== 新增验证集，假设放在 'rubik_val_shards' 目录 ======
    val_dataset = RubikDataset(data_dir='rubik_val_shards', max_files=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # 2. Model
    model = RubikSeq2SeqTransformer(num_layers=4, d_model=128)
    model = model.to(device)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 调度器 (LinearWarmupCosineAnnealingLR)
    warmup_epochs = 50
    max_epochs = 1000
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs
    )

    # 4. Training loop
    epochs = 1000
    best_val_acc = 0.0  # 记录验证集准确率的最高值

    for epoch in range(1, epochs+1):
        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")

        val_acc = evaluate_seq2seq_accuracy(model, val_loader, device)
        print(f"[Validation] Epoch {epoch}, Val_Acc={val_acc:.4f}")

        # 每 50 个 epoch 做一次验证
        if epoch % 50 == 0:

            # 保存当前 epoch 的模型
            ckpt_path = f"rubik_model_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"已保存模型到 {ckpt_path}")

            # 如果比最优准确率更高，则更新 best 并另存一份
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "rubik_model_best.pth")
                print(f"当前准确率最好 ({val_acc:.4f})，已更新 rubik_model_best.pth")

    # 最后再保存一次 (可选)
    torch.save(model.state_dict(), "rubik_model_final.pth")
    print("训练结束，已保存最终模型为 rubik_model_final.pth")


if __name__ == "__main__":
    main()
