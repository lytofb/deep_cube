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
# 在文件开头（例如在 import 部分后）添加：
import torch.nn.init as init


from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

scaler = GradScaler()

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

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

        # 使用混合后的输入进行前向传播，计算最终 loss
        with autocast():
            logits = model(src, decoder_input)  # (B, seq_len-1, num_moves)
            loss = criterion(logits.view(-1, logits.size(-1)), target_output.contiguous().view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * tgt.size(0)

        # logits = logits.reshape(-1, num_moves)       # => (B*(seq_len-1), num_moves)
        # target_output = target_output.reshape(-1)    # => (B*(seq_len-1))
        #
        # loss = criterion(logits, target_output)
        # loss.backward()
        # optimizer.step()
        #
        # total_loss += loss.item() * src.size(0)

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

        # 使用混合后的输入进行前向传播，计算最终 loss
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

    # 新增：初始化 Comet ML 实验
    # 初始化 Comet ML 实验（使用 config 中的参数）
    experiment = start(
      api_key=config.comet.api_key,
      project_name=config.comet.project_name,
      workspace=config.comet.workspace
    )
    # 记录所有超参数
    experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir=config.data.train_dir, num_samples=config.data.num_samples, max_files=None)
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

    # ====== 新增验证集，假设放在 'rubik_val_shards' 目录 ======
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
    # Model：使用配置中定义的模型参数
    model = RubikSeq2SeqTransformer(
        num_layers=config.model.num_layers,
        d_model=config.model.d_model,
        input_dim=config.model.input_dim,
        nhead=config.model.nhead,
        num_moves=config.model.num_moves,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    model.apply(init_weights)  # 新增：应用 He 初始化
    # 如果有多张 GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行 DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    log_model(experiment, model=model, model_name="TheModel")

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs
    )

    # 4. Training loop
    epochs = config.train.max_epochs
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

        experiment.log_metric("train_loss", avg_loss, step=epoch)
        experiment.log_metric("lr", current_lr, step=epoch)
        experiment.log_metric("val_accuracy", val_acc, step=epoch)

    # 最后再保存一次 (可选)
    torch.save(model.state_dict(), "rubik_model_final.pth")
    print("训练结束，已保存最终模型为 rubik_model_final.pth")


def main_ddp():
    """
    DDP 多卡训练入口函数
    """
    # 获取 local_rank
    local_rank = int(os.environ["LOCAL_RANK"])

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    experiment = start(
      api_key=config.comet.api_key,
      project_name=config.comet.project_name,
      workspace=config.comet.workspace
    )

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir=config.data.train_dir, num_samples=config.data.num_samples, max_files=None)

    # ====== 新增验证集，假设放在 'rubik_val_shards' 目录 ======
    val_dataset = RubikDataset(data_dir=config.data.val_dir, max_files=None)

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,  # 注意，这里需要 False，分布式时用 sampler 控制 shuffle
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
    model.apply(init_weights)  # 新增：应用 He 初始化
    model = model.to(device)

    # 用 DDP 包装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs
    )

    # 4. Training loop
    epochs = config.train.max_epochs
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 分布式训练时，每个 epoch 都要在 sampler 上设置一下随机种子
        train_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 只让 rank=0 的进程打印或做验证/保存模型
        if dist.get_rank() == 0:
            print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")
            val_acc = evaluate_seq2seq_accuracy(model, val_loader, device)
            print(f"[Validation] Epoch {epoch}, Val_Acc={val_acc:.4f}")
            # ... 这里也可以做一些 if epoch % 50 == 0: 保存模型 的操作 ...

    # 结束
    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()

