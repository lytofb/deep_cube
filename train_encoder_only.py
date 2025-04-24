# train.py

from comet_ml import start
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from sampler.distributed_weight_sampler import DistributedWeightedSampler
from utils import PAD_TOKEN, EOS_TOKEN, SOS_TOKEN, FocalLoss, VOCAB_SIZE

from inference import iterative_greedy_decode_seq2seq, random_scramble_cube, beam_search

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
# 在文件开头（例如在 import 部分后）添加：
import torch.nn.init as init


from dataset_rubik import RubikDataset, collate_fn
from models.model_encoder_only import RubikEncoderOnly

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

scaler = GradScaler()
use_amp = config.train.get("use_amp", True)

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
        target_output = tgt[:, 1]   # (B,)

        # 使用混合后的输入进行前向传播，计算最终 loss
        with autocast(enabled=use_amp):
            logits = model(src)  # (B, num_moves)
            loss = criterion(logits, target_output)

        scaler.scale(loss).backward()
        # 先反缩放梯度
        scaler.unscale_(optimizer)
        # 执行梯度裁剪，clip_value 可根据需要调整（例如 1.0）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        target_output = tgt[:, 1]   # (B,)

        logits = model(src)  # => (B, num_moves)
        # 取 argmax => (B, seq_len-1)
        pred_tokens = logits.argmax(dim=-1)     # (B,)

        # 统计预测正确的数量
        mask = target_output != PAD_TOKEN
        correct = (pred_tokens == target_output) & mask
        total_correct += correct.sum().item()
        total_count += mask.sum().item()

    if total_count == 0:
        return 0.0
    return total_correct / total_count

def create_collate_fn_random_trunc(max_history_len: int, pad_token: int = PAD_TOKEN):
    """
    返回一个 collate_fn 函数，用于在批处理中对 src 序列随机截断到 [0, max_history_len]，
    并对不足部分填充 PAD。

    Args:
        max_history_len: 截断后的最大长度，如 8
        pad_token: 用于填充的 token 索引

    Returns:
        collate_fn_random_trunc: 一个可供 DataLoader 调用的函数
    """

    def collate_fn_random_trunc(batch):
        """
        batch: List of (src, tgt)
               - src: shape (src_seq_len, input_dim) 或者 (src_seq_len,) 的索引
               - tgt: shape (tgt_seq_len,) 的索引
        """
        batch_src = []
        batch_tgt = []

        for src_item, tgt_item in batch:
            # ---------- 随机截断 ----------
            actual_len = src_item.shape[0]  # 如果 src_item 是二维 (seq_len, input_dim)，就取 seq_len
            k = random.randint(0, max_history_len)  # 在 [0, max_history_len] 范围内随机
            k = min(k, actual_len)  # 确保不会超过实际长度

            truncated_src = src_item[:k]  # 截断

            # ---------- Pad 到固定长度 max_history_len ----------
            pad_size = max_history_len - k
            if pad_size > 0:
                # 如果 src_item是 (seq_len, input_dim)，则需要构造一个 (pad_size, input_dim) 的填充
                # 并只在“最后一列/某一列”标注 pad_token；下面是示意写法，需根据你实际格式调整
                pad_part = torch.zeros((pad_size, truncated_src.shape[1]), dtype=truncated_src.dtype)
                # 假设最后一列是 move token，才需要填 pad_token
                pad_part[:, -1] = pad_token
                truncated_src = torch.cat([truncated_src, pad_part], dim=0)

            # 对 tgt_item 是否也要随机截断，看你需求而定，下面示例保留原样
            # 如果需要对 tgt_item 做 pad 到固定长度，也可参照类似写法
            batch_src.append(truncated_src)
            batch_tgt.append(tgt_item)

        # ---------- 组装 batch ----------
        batch_src = torch.stack(batch_src, dim=0)  # (B, max_history_len, input_dim)
        # 假设 tgt_item 是 1D token 序列，不同长度 => pad_sequence 统一长度
        batch_tgt = torch.nn.utils.rnn.pad_sequence(
            batch_tgt, batch_first=True, padding_value=pad_token
        )

        return batch_src, batch_tgt

    return collate_fn_random_trunc


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

    if dist.get_rank() == 0:
        experiment = start(
          api_key=config.comet.api_key,
          project_name=config.comet.project_name,
          workspace=config.comet.workspace
        )
        experiment.log_parameters(OmegaConf.to_container(config, resolve=True))
        experiment.log_asset(file_data="models/model_history_transformer.py")
        experiment.log_asset(file_data="dataset_rubik.py")

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir=config.data.train_dir,
                                 history_len=config.data.max_history_len,
                                 num_samples=config.data.num_samples,
                                 max_files=None)

    # ====== 新增验证集，假设放在 'rubik_val_shards' 目录 ======
    val_dataset = RubikDataset(data_dir=config.data.val_dir,
                               history_len=config.data.max_history_len,
                               max_files=None)

    # ---------- 1. 统计首步频次并计算倒频率 ----------
    first_counts = torch.bincount(
        torch.tensor(train_dataset.first_moves),
        minlength=VOCAB_SIZE
    ).double()                      # (C,)

    inv_freq = 1.0 / (first_counts + 1e-8)   # 避免除 0
    weights   = inv_freq[torch.tensor(train_dataset.first_moves)]  # (N,)

    # ---------- 2. 使用自定义 DistributedWeightedSampler ----------
    train_sampler = DistributedWeightedSampler(weights)
    # 使用 DistributedSampler
    # train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # 1) 先根据 config 创建一个 collate_fn
    my_collate_fn_random_trunc = create_collate_fn_random_trunc(
        max_history_len=config.data.max_history_len,  # 例如 8
        pad_token=PAD_TOKEN  # 填充截断的TOKEN
    )

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
    model = RubikEncoderOnly(
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
    if dist.get_rank() == 0:
        log_model(experiment, model=model, model_name="TheModel")


    # 3. Optimizer & Loss
    criterion = FocalLoss(ignore_index=PAD_TOKEN)
    # criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = model.configure_optimizers(learning_rate=config.train.learning_rate,weight_decay=config.train.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    eta_min = 1/3*config.train.learning_rate
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.train.warmup_epochs,
        max_epochs=config.train.max_epochs,
        eta_min=eta_min
    )

    # 用 DDP 包装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Training loop
    epochs = config.train.max_epochs
    if dist.get_rank() == 0:
        early_stop_patience = config.train.early_stop_patience  # 可根据需要调整 patience，比如 5 个 epoch
        epochs_without_improvement = 0

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
            # 请在 "if epoch % 10 == 0:" 的 pass 替换为以下内容

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), "rubik_model_best.pth")
                print(f"当前准确率最好 ({val_acc:.4f})，已更新 rubik_model_best.pth")
            else:
                epochs_without_improvement += 1
                print(f"没有改进。连续 {epochs_without_improvement} 个 epoch 无提升")

            if epochs_without_improvement >= early_stop_patience:
                print("早停条件满足，停止训练。")
                break

            # if epoch % 2 == 0:
            #     print(f"===== Free Run Evaluate at Epoch {epoch} =====")
            #     model.eval()
            #     with torch.no_grad():
            #         # 演示: 随机打乱一个魔方，调用 iterative_greedy_decode_seq2seq 来做推理
            #         cube, scramble_moves = random_scramble_cube(steps=8)  # 你也可改为固定打乱
            #         print("Scramble moves:", scramble_moves)
            #
            #         # 进行自由推断
            #         pred_tokens = beam_search(
            #             model=model,
            #             cube=cube,
            #             history_len=config.data.max_history_len,  # 例如8，与训练时一致
            #             max_len=50,
            #             device=device
            #         )
            #
            #         print("Predicted token IDs:", pred_tokens)
            #
            #     model.train()

            # 每 20 个 epoch 做一次验证
            if epoch % 20 == 0:

                # 保存当前 epoch 的模型
                ckpt_path = f"rubik_model_epoch{epoch}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"已保存模型到 {ckpt_path}")

            experiment.log_metric("train_loss", avg_loss, step=epoch)
            experiment.log_metric("lr", current_lr, step=epoch)
            experiment.log_metric("val_accuracy", val_acc, step=epoch)

    # 最后再保存一次 (可选)
    torch.save(model.state_dict(), "rubik_model_final.pth")
    print("训练结束，已保存最终模型为 rubik_model_final.pth")
    # 结束
    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()

