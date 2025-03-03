# train.py

from comet_ml import start
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 可以根据需要开启/关闭 profiler
# from torch.profiler import profile, record_function, ProfilerActivity

from dataset_rubik_pact import RubikDatasetPACT, collate_fn

# ======= 新增：我们自己定义的 PACT GPT 模型 (或从另一文件中导入) =======
from models.model_pact_transformer import RubikGPT

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf

config = OmegaConf.load("configtest.yaml")

scaler = GradScaler()


def train_one_epoch_pact(model, dataloader, optimizer, criterion, device, epoch):
    """
    不再使用外部 tgt，而是从 src 的第 54 维提取标签
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for src,_ in tqdm(dataloader, desc=f"Training (epoch={epoch})"):
        # 假设 dataloader 现在只返回 src
        # 如果你原本 dataset 还有 (src, tgt)，就改成只返回 src 或忽略 tgt
        src = src.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(src)  # => (B, 2T, vocab_size)

            # 只取 action token => (B, T, vocab_size)
            action_logits = logits[:, 1::2, :]

            # 从 src[..., 54] 拿标签
            # 假设它已在数据集里存的是正确的动作索引 (int)
            label_from_src = src[:, :, 54].long()  # => (B, T)

            loss = criterion(
                action_logits.transpose(1, 2),  # => (B, vocab_size, T)
                label_from_src                 # => (B, T)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = src.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_pact_accuracy(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_count = 0

    for src,_ in dataloader:
        src = src.to(device)

        logits = model(src)  # => (B,2T,vocab_size)
        action_logits = logits[:, 1::2, :]  # => (B,T,vocab_size)

        # 从 src 拿到标签
        label_from_src = src[:, :, 54].long()  # => (B, T)

        # argmax => (B,T)
        pred_tokens = action_logits.argmax(dim=-1)

        total_correct += (pred_tokens == label_from_src).sum().item()
        total_count += label_from_src.numel()

    return total_correct / total_count if total_count else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 1. Comet ML 初始化 --
    experiment = start(
        api_key=config.comet.api_key,
        project_name=config.comet.project_name,
        workspace=config.comet.workspace
    )
    experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    # -- 2. Dataset & Dataloader --
    train_dataset = RubikDatasetPACT(data_dir=config.data.train_dir,
                                 num_samples=config.data.num_samples,
                                 max_files=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.train.prefetch_factor
    )

    val_dataset = RubikDatasetPACT(data_dir=config.data.val_dir, num_samples=config.data.num_samples, max_files=None)
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

    # -- 3. 构建 PACT GPT 模型 --
    model = RubikGPT(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        max_seq_len=config.model.max_seq_len,
        ff_dim=config.model.d_model * 4,  # 例如 feedforward = 4*d_model
        dropout=config.model.dropout,
        vocab_size=config.model.num_moves  # 假设要预测的动作数
    )
    model = model.to(device)
    log_model(experiment, model=model, model_name="PACT_RubikModel")

    # -- 4. 优化器 & 损失 & 学习率调度 --
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs
    )

    # -- 5. 训练循环 --
    epochs = config.train.max_epochs
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch_pact(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")
        val_acc = evaluate_pact_accuracy(model, val_loader, device)
        print(f"[Validation] Epoch {epoch}, Val_Acc={val_acc:.4f}")

        # 每 N 个 epoch 保存一次
        if epoch % 50 == 0:
            ckpt_path = f"rubik_model_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"已保存模型到 {ckpt_path}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "rubik_model_best.pth")
                print(f"当前准确率最好 ({val_acc:.4f})，已更新 rubik_model_best.pth")

        # Comet 日志
        experiment.log_metric("train_loss", avg_loss, step=epoch)
        experiment.log_metric("lr", current_lr, step=epoch)
        experiment.log_metric("val_accuracy", val_acc, step=epoch)

    torch.save(model.state_dict(), "rubik_model_final.pth")
    print("训练结束，已保存最终模型为 rubik_model_final.pth")


if __name__ == "__main__":
    main()
