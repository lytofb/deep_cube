# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

from dataset_rubik import RubikDataset, collate_fn
from models.model_history_transformer import RubikSeq2SeqTransformer

from utils.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR


# 或者 from models.model_cnn import RubikCNN
# 或者 from models.model_transformer import RubikTransformer

def train_one_epoch_old(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for states, moves in dataloader:
        # states shape=(B,54), moves shape=(B,)
        states = states.to(device)
        moves = moves.to(device)

        optimizer.zero_grad()
        logits = model(states)  # (B,18)
        loss = criterion(logits, moves)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * states.size(0)

    return total_loss / len(dataloader.dataset)

def train_one_epoch_history(model, dataloader, optimizer, criterion, device):
    """
    针对使用RubikHistoryTransformer的训练循环：
    - seqs 形状: (B, seq_len, 55)
    - labels 形状: (B,)
    """
    model.train()
    total_loss = 0.0

    for seqs, labels in tqdm(dataloader, desc="Training"):
        # seqs => (B, seq_len, 55)
        # labels => (B,)

        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        # 前向
        logits = model(seqs)  #  => (B, num_moves)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 统计Loss
        total_loss += loss.item() * seqs.size(0)

    # 返回平均Loss
    return total_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir='rubik_shards', max_files=None)  # or some subset
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,  # 开启 pin_memory
        persistent_workers=True,  # 若 PyTorch 版本支持，保持worker进程常驻，避免每个 epoch 重启开销
        prefetch_factor=4  # 根据具体 CPU 核数和内存情况，可适当调整预取因子
    )

    # 2. Model
    model = RubikSeq2SeqTransformer(num_layers=24,d_model=256)
    model = model.to(device)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 新增：使用LinearWarmupCosineAnnealingLR调度器
    # 注意：max_epochs需要大于或等于warmup_epochs
    warmup_epochs = 5
    max_epochs = 50  # 根据实际训练的总epoch数设置
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)

    # 4. Training loop
    epochs = 1000
    for epoch in range(epochs):
        avg_loss = train_one_epoch_history(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")
        # with profile(
        #         activities=[
        #             ProfilerActivity.CPU,
        #             ProfilerActivity.CUDA
        #         ],
        #         record_shapes=True,  # 记录张量维度信息
        #         profile_memory=True,  # 记录显存/内存使用
        #         with_stack=False  # 若要记录Python调用栈可设True
        # ) as prof:
        #     avg_loss = train_one_epoch_history(model, train_loader, optimizer, criterion, device)
        #     scheduler.step()
        #     current_lr = scheduler.get_last_lr()[0]
        #     print(f"Epoch {epoch}, Loss={avg_loss:.4f}, LR={current_lr:.6f}")
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")
    # 5. 保存模型
    torch.save(model.state_dict(), "rubik_model.pth")
    print("模型已保存到 rubik_model.pth")


if __name__ == "__main__":
    main()
