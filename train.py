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


def train_one_epoch_seq2seq(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc="Training"):
        # src: (B, src_seq_len, 55)，tgt: (B, tgt_seq_len)
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()

        # teacher forcing：外部切片
        decoder_input = tgt[:, :-1]  # (B, tgt_seq_len - 1)
        target_output = tgt[:, 1:]  # (B, tgt_seq_len - 1)

        # 前向传播，把 decoder_input 传给模型
        logits = model(src, decoder_input)  # => (B, tgt_seq_len-1, num_moves)

        # 展平计算损失
        B, seq_len, num_moves = logits.shape
        logits = logits.reshape(-1, num_moves)  # => (B*(seq_len-1), vocab_size)
        target_output = target_output.reshape(-1)  # => (B*(seq_len-1))

        loss = criterion(logits, target_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & DataLoader
    train_dataset = RubikDataset(data_dir='rubik_shards', max_files=None)  # or some subset
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,  # 开启 pin_memory
        persistent_workers=True,  # 若 PyTorch 版本支持，保持worker进程常驻，避免每个 epoch 重启开销
        prefetch_factor=4  # 根据具体 CPU 核数和内存情况，可适当调整预取因子
    )

    # 2. Model
    # model = RubikSeq2SeqTransformer(num_layers=4,d_model=2048)
    model = RubikSeq2SeqTransformer(num_layers=4,d_model=2048)
    model = model.to(device)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 新增：使用LinearWarmupCosineAnnealingLR调度器
    # 注意：max_epochs需要大于或等于warmup_epochs
    warmup_epochs = 50
    max_epochs = 1000  # 根据实际训练的总epoch数设置
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)

    # 4. Training loop
    epochs = 1000
    for epoch in range(epochs):
        avg_loss = train_one_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
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
