# train.py

from comet_ml import start
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer

from utilsp.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler

from omegaconf import OmegaConf
config = OmegaConf.load("config.yaml")

scaler = GradScaler()


def train_one_epoch_seq2seq(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    #(model, train_loader, optimizer, criterion, device)
    """
    用“单步策略+Scheduled Sampling”来训练。

    - model: 虽然还是RubikSeq2SeqTransformer，但我们这里当做“policy”用，逐步解码。
    - 每个 batch 的 (src, tgt)，其中:
        src: (B, src_seq_len, 55)，一般表示初始或其它必要信息。
        tgt: (B, tgt_seq_len)，序列动作。tgt[:, 0] 通常是SOS_TOKEN，tgt[:, 1:]是真实动作。
    - 我们不再一次性把 tgt[:, :-1] 直接喂到 Decoder，
      而是 step by step 地喂，并在每一步根据一定概率决定用“真值动作”或“模型预测动作”。

    """
    model.train()
    total_loss = 0.0

    # 定义一个“使用模型自身预测”的概率（即Scheduled Sampling的概率）
    # 你可以根据 epoch 做一个递增或递减策略，这里仅作示例
    sampling_prob = min(0.5, epoch / total_epochs * 0.5)
    # 例如：前期主要用 teacher forcing，后期逐渐使用更多模型预测

    for src, tgt in tqdm(dataloader, desc=f"Training (epoch={epoch})"):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        batch_size, seq_len = tgt.shape

        # 我们要“逐步”产生decoder输入，并在每一步计算loss
        # decoder_input 初始化只包含 tgt[:, 0] (一般是SOS_TOKEN)
        decoder_input = tgt[:, 0].unsqueeze(1)  # shape: (B, 1)

        # 用来收集每个时间步的预测结果（logits），最后一起计算loss
        all_step_logits = []

        optimizer.zero_grad()

        # 逐步解码 seq_len - 1 步（因为第一步是SOS，不需要预测）
        for t in range(seq_len - 1):
            # 前向传播：把当前decoder_input喂给模型
            with autocast():
                # logits_current: (B, decoder_input_len, num_moves)
                logits_current = model(src, decoder_input)
                # 我们只关心最后一个时间步的输出 => (B, num_moves)
                last_step_logits = logits_current[:, -1, :]

            # 把这个时刻的预测结果存起来
            all_step_logits.append(last_step_logits)

            # 根据一定概率决定下一个时间步要不要用模型预测值
            # 先取出当前的预测token:
            pred_tokens = last_step_logits.argmax(dim=-1)  # shape: (B,)

            # gt_tokens = tgt[:, t+1] 是真实的下一步动作
            gt_tokens = tgt[:, t + 1]

            # 随机mask判断是否使用模型预测值
            # 用 sampling_prob 来控制“使用模型预测”的概率
            coin_toss = torch.rand(batch_size, device=device)  # (B,)
            use_model_prediction = (coin_toss < sampling_prob)

            # 组装“下一步 decoder 输入”，如果采样就用pred，否则用真值
            next_tokens = torch.where(use_model_prediction, pred_tokens, gt_tokens)
            # 拼到decoder_input后面
            next_tokens = next_tokens.unsqueeze(1)  # (B,1)
            decoder_input = torch.cat([decoder_input, next_tokens], dim=1)

        # 最后把收集的所有时刻预测对齐 target => (B, seq_len-1)
        # stack => (B, seq_len-1, num_moves)
        all_step_logits = torch.stack(all_step_logits, dim=1)
        # reshape
        all_step_logits = all_step_logits.view(-1, all_step_logits.size(-1))  # => (B*(seq_len-1), num_moves)

        target_tokens = tgt[:, 1:].contiguous().view(-1)  # => (B*(seq_len-1),)

        with autocast():
            loss = criterion(all_step_logits, target_tokens)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size

    # 返回平均loss
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
    train_dataset = RubikDataset(data_dir=config.data.train_dir, max_files=None)
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
        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device ,epoch ,epochs)
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


if __name__ == "__main__":
    main()
