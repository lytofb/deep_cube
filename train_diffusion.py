# train_diffusion.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1) 引入新的离散数据集 (已在内存生成或从pkl加载)
from dataset_rubik_seq import RubikSeqDataset, collate_fn
from models.model_diffusion_lm import DiscreteDiffusionLM,DiffusionLMTransformer

from utils import (
    COLOR_CHARS,
    MOVES_POOL,
    MOVE_TO_IDX,
    PAD_TOKEN,
    MASK_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    VOCAB_SIZE,
    convert_state_to_tensor,
    move_str_to_idx,
)

def train_diffusion_lm_condition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset
    ds_inmem = RubikSeqDataset(num_samples=10000, min_scramble=8, max_scramble=25)
    print("Dataset size:", len(ds_inmem))

    dl = DataLoader(ds_inmem, batch_size=128, shuffle=True, collate_fn=collate_fn)

    # 2) Diffusion + 模型
    diffusion = DiscreteDiffusionLM(
        mask_id=MASK_TOKEN,        # 仅用于噪声
        schedule_start=0.1,
        schedule_end=0.7,
        num_steps=10
    )
    model = DiffusionLMTransformer(
        vocab_size=VOCAB_SIZE,     # 22
        cond_dim=54,
        d_model=128,
        n_layers=4,
        n_heads=4,
        p_drop=0.1,
        max_seq_len=50
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # CE时, ignore_index=PAD_TOKEN=18, 使得填充区不计入loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    epochs = 5
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for conds, seqs, lengths in dl:
            conds = conds.to(device)  # (B,54)
            seqs = seqs.to(device)    # (B,L)
            B,L = seqs.shape

            # x0 = seqs
            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            x_t = diffusion.q_sample(seqs, t)   # 替换为MASK_TOKEN=19

            logits = model(x_t, conds)  # (B,L,22)
            loss = criterion(logits.view(-1, VOCAB_SIZE), seqs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*B
            total_count += B

        avg_loss = total_loss / total_count if total_count>0 else 0.0
        print(f"Epoch {ep}, Loss={avg_loss:.4f}")


if __name__ == "__main__":
    train_diffusion_lm_condition()
