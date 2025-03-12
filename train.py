# train.py

from comet_ml import start
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

# 更新数据集导入路径，使用重构后的 dataset.py
from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

scaler = GradScaler()


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
        with autocast():
            logits = model(src, decoder_input)  # => (B, tgt_seq_len-1, num_moves)
            loss = criterion(logits.view(-1, logits.size(-1)), target_output.contiguous().view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * tgt.size(0)

    return total_loss / len(dataloader.dataset)


def train_one_epoch_seq2seq_mix(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    # 设置 scheduled sampling 的概率（例如：前期主要用 teacher forcing，后期逐渐使用更多模型预测）
    sampling_prob = min(0.5, epoch / total_epochs * 0.5)

    for src, tgt in tqdm(dataloader, desc=f"Training (epoch={epoch})"):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # 构造 teacher forcing 下的 decoder 输入与目标
        decoder_input = tgt[:, :-1].clone()   # (B, seq_len-1)
        target_tokens = tgt[:, 1:].clone()      # (B, seq_len-1)

        optimizer.zero_grad()

        # 先做一次前向传播（不计算梯度），得到基于 teacher forcing 的预测，用于 token-level mixing
        with torch.no_grad():
            teacher_logits = model(src, decoder_input)
            teacher_preds = teacher_logits.argmax(dim=-1)  # (B, seq_len-1)

        # 对 decoder 输入的每个 token（除第一个 token 外）随机决定是否替换为模型预测
        mix_mask = (torch.rand(decoder_input.shape, device=device) < sampling_prob)
        mix_mask[:, 0] = False  # 保持 SOS token 不变
        mixed_decoder_input = torch.where(mix_mask, teacher_preds, decoder_input)

        with autocast():
            logits = model(src, mixed_decoder_input)  # (B, seq_len-1, num_moves)
            loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.contiguous().view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * tgt.size(0)

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

        decoder_input = tgt[:, :-1]
        target_output = tgt[:, 1:]  # 形状 (B, seq_len-1)

        logits = model(src, decoder_input)  # => (B, seq_len-1, num_moves)
        pred_tokens = logits.argmax(dim=-1)

        total_correct += (pred_tokens == target_output).sum().item()
        total_count += target_output.numel()

    return total_correct / total_count if total_count > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 Comet ML 实验
    experiment = start(
      api_key=config.comet.api_key,
      project_name=config.comet.project_name,
      workspace=config.comet.workspace
    )
    experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(
        data_dir=config.data.train_dir,
        num_samples=config.data.num_samples,
        max_files=None
    )
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

    # 新增验证集（假设放在 config.data.val_dir）
    val_dataset = RubikDataset(data_dir=config.data.val_dir, max_files=None)
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

    # 2. Model
    model = RubikSeq2SeqTransformer(
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        input_dim=config.model.input_dim,
        nhead=config.model.nhead,
        num_moves=config.model.num_moves,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行 DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    log_model(experiment, model=model, model_name="TheModel")

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs
    )

    # 4. Training loop
    epochs = config.train.max_epochs
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")

        val_acc = evaluate_seq2seq_accuracy(model, val_loader, device)
        print(f"[Validation] Epoch {epoch}, Val_Acc={val_acc:.4f}")

        if epoch % 50 == 0:
            ckpt_path = f"rubik_model_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"已保存模型到 {ckpt_path}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "rubik_model_best.pth")
                print(f"当前准确率最好 ({val_acc:.4f})，已更新 rubik_model_best.pth")

        experiment.log_metric("train_loss", avg_loss, step=epoch)
        experiment.log_metric("lr", current_lr, step=epoch)
        experiment.log_metric("val_accuracy", val_acc, step=epoch)

    torch.save(model.state_dict(), "rubik_model_final.pth")
    print("训练结束，已保存最终模型为 rubik_model_final.pth")


def main_ddp():
    """
    DDP 多卡训练入口函数
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    experiment = start(
      api_key=config.comet.api_key,
      project_name=config.comet.project_name,
      workspace=config.comet.workspace
    )

    train_dataset = RubikDataset(
        data_dir=config.data.train_dir,
        num_samples=config.data.num_samples,
        max_files=None
    )
    val_dataset = RubikDataset(data_dir=config.data.val_dir, max_files=None)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.train.prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.train.prefetch_factor
    )

    model = RubikSeq2SeqTransformer(
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        input_dim=config.model.input_dim,
        nhead=config.model.nhead,
        num_moves=config.model.num_moves,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs
    )

    epochs = config.train.max_epochs
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if dist.get_rank() == 0:
            print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")
            val_acc = evaluate_seq2seq_accuracy(model, val_loader, device)
            print(f"[Validation] Epoch {epoch}, Val_Acc={val_acc:.4f}")
            # 可在此处添加模型保存逻辑

    dist.destroy_process_group()


if __name__ == "__main__":
    # 若使用单卡/单机训练，则调用 main()，多卡训练则调用 main_ddp()
    if "LOCAL_RANK" in os.environ:
        main_ddp()
    else:
        main()
