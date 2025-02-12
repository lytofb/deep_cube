# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_rubik import RubikDataset, collate_fn
# from models.model_transformer import RubikTransformer
from models.model_history_transformer import RubikHistoryTransformer


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

        seqs = seqs.to(device)
        labels = labels.to(device)

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # 2. Model
    model = RubikHistoryTransformer(num_layers=24,d_model=256)
    model = model.to(device)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training loop
    epochs = 1
    for epoch in range(epochs):
        avg_loss = train_one_epoch_history(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Loss={avg_loss:.4f}")

    # 5. 保存模型
    torch.save(model.state_dict(), "rubik_model.pth")
    print("模型已保存到 rubik_model.pth")


if __name__ == "__main__":
    main()
