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
from utils import PAD_TOKEN

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

    for src,tgt in tqdm(dataloader, desc=f"Training (epoch={epoch})"):
        # 假设 dataloader 现在只返回 src
        # 如果你原本 dataset 还有 (src, tgt)，就改成只返回 src 或忽略 tgt
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(src)  # => (B, 2T, vocab_size)

            # 只取 action token => (B, T, vocab_size)
            action_logits = logits[:, 1::2, :]

            loss = criterion(
                action_logits.transpose(1, 2),  # => (B, vocab_size, T)
                tgt                 # => (B, T)
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

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src)  # => (B,2T,vocab_size)
        action_logits = logits[:, 1::2, :]  # => (B,T,vocab_size)

        # argmax => (B,T)
        pred_tokens = action_logits.argmax(dim=-1)

        total_correct += (pred_tokens == tgt).sum().item()
        total_count += tgt.numel()

    return total_correct / total_count if total_count else 0.0


@torch.no_grad()
def evaluate_pact_freerun(model, dataloader, device, max_steps=50):
    """
    对验证集做 “free run”（多步自回归） 推断并计算 token-level Accuracy。
    与 teacher forcing 不同，这里在每一步都使用模型上一步的预测动作，更新到输入序列再前向。

    Args:
        model: 你的 PACT/GPT 模型
        dataloader: 验证集 DataLoader，返回 (src_seq, tgt_seq)。
                    - src_seq: shape=(B, seq_len, 55)，其中每行=[state, action]
                    - tgt_seq: shape=(B, seq_len)，与 src_seq 对齐 (向左 shift 一位再 +EOS)，或者你自己的定义
        device: 训练使用的device
        max_steps: free-run 的最大步数限制

    Returns:
        accuracy: (float) 多步推理下的动作预测准确率
    """
    model.eval()
    total_correct = 0
    total_count = 0

    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
        # src_batch: (B, seq_len, 55)
        # tgt_batch: (B, seq_len)   (如果你是向左 shift + EOS 的话)

        src_batch = src_batch.to(device)  # (B, seq_len, 55)
        tgt_batch = tgt_batch.to(device)  # (B, seq_len)

        B, seq_len, _ = src_batch.shape

        # 对于每个样本，做 free-run
        for b_i in range(B):
            # 取单个样本
            src_seq = src_batch[b_i]  # shape=(seq_len, 55)
            tgt_seq = tgt_batch[b_i]  # shape=(seq_len,)

            # 我们假设 “src_seq[i] = (state_i, action_i)”，其中 action_i 最开始是训练中的记录
            # 但 free-run 要自回归，所以要在每个 step 把 "上一步预测的 action" 写回 src_seq[i,54]

            # 先复制一份到 CPU/GPU Tensor (1, step, 55)
            # 也可放 list 里，每次 append
            # 这里演示“每次只加一对 (state, action=pred)”，就像你在 "greedy_decode_pact" 那样做
            seq_list = []
            # 第 0 步: 把初始 (state_0, action_0) 先放进去
            # 如果你想从 “action=-1 or PAD” 开始，也可改
            seq_list.append(src_seq[0].clone().unsqueeze(0))  # shape=(1,55)

            predicted_actions = []  # 记录该样本多步产生的动作
            gold_actions = []  # 记录真实标签 (tgt_seq 的)

            # free-run 循环，直到 seq_len-1 或 max_steps
            # seq_len-1 是因为最后一个位置 typically 对应 EOS
            # 你也可视需求让它跑到 seq_len
            run_steps = min(seq_len, max_steps)

            for step in range(run_steps):
                # 把 seq_list 合并 => (cur_T, 55)，再加个 batch_dim => (1, cur_T, 55)
                cur_src = torch.cat(seq_list, dim=0).unsqueeze(0).to(device)  # => (1, cur_T, 55)
                cur_T = cur_src.size(1)

                # 前向 => (1, 2*cur_T, vocab_size)
                logits = model(cur_src)
                # 取最后一个 action token => 索引= 2*cur_T -1
                last_action_logits = logits[:, 2 * cur_T - 1, :]  # => (1, vocab_size)

                pred_action = torch.argmax(last_action_logits, dim=-1).item()
                predicted_actions.append(pred_action)

                # “真实动作”如果按 “tgt_seq[step]” 对应 “src_seq[step, 54]” 的话:
                gold_action = tgt_seq[step].item()
                gold_actions.append(gold_action)

                # 把 predicted_action 写回 “seq_list[-1][54]” => 让它成为“本步的真实 action”
                # 这样 2T tokens 里第 (2*cur_T -1) 就是 pred_action
                # 但 GPT 的下个输入 token = (state_{next}, action=PAD) 通常
                #   => 需要新的 (55,) : [ new_state, PAD ]
                #   这里如果 dataset 里 state_{step+1} 就在 src_seq[step+1, :54]
                #   所以可以先“更新 seq_list[-1][54] = pred_action”，然后 append 下一个 (state_{step+1}, action=PAD)
                seq_list[-1][0, 54] = pred_action  # 把最后 action 改成 pred

                # 如果 step+1 < seq_len，则追加下一条 (state_{step+1}, action=?)
                if (step + 1) < seq_len:
                    # new_sa = (state_{step+1}, action=PAD?)
                    new_sa = src_seq[step + 1].clone()
                    # 先把 new_sa[54]=PAD 或 -1
                    # 也可以保留 data 里的 move，这要看你训练时的形式
                    new_sa[54] = -1  # or PAD_TOKEN
                    seq_list.append(new_sa.unsqueeze(0))  # shape=(1,55)

                # 如果 pred_action == EOS_TOKEN 或 PAD_TOKEN，假设你想提前结束
                #   => break
                # if pred_action in [EOS_TOKEN, PAD_TOKEN]:
                #     break

            # 统计该样本的 token-level 正确率
            # predicted_actions[i] vs. gold_actions[i]
            # 有可能 gold_actions 里最后有 EOS，你可以先不统计 EOS
            steps_count = len(predicted_actions)
            for i in range(steps_count):
                if gold_actions[i] != -1 and gold_actions[i] != PAD_TOKEN:
                    if predicted_actions[i] == gold_actions[i]:
                        total_correct += 1
                    total_count += 1

    if total_count == 0:
        return 0.0
    return total_correct / total_count


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

            # 2) 再做 free-run 验证
            val_acc_freerun = evaluate_pact_freerun(model, val_loader, device)
            print(f"[Validation Free-Run] Epoch={epoch}, Acc={val_acc_freerun:.4f}")

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
