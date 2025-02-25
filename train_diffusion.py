from comet_ml import start
from comet_ml.integration.pytorch import log_model
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1) 引入新的离散数据集 (已在内存生成或从pkl加载)
from dataset_rubik_seq import RubikSeqDataset, collate_fn
from models.model_diffusion_lm import DiscreteDiffusionLM, DiffusionLMTransformer

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
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置文件
    config = OmegaConf.load("config_diff_lm.yaml")

    # 初始化 Comet ML 实验
    experiment = start(
        api_key=config.comet.api_key,
        project_name=config.comet.project_name,
        workspace=config.comet.workspace
    )
    experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    # 1) Dataset，根据配置文件中的参数初始化数据集
    ds_inmem = RubikSeqDataset(
        num_samples=config.data.num_samples,
        min_scramble=config.data.min_scramble,
        max_scramble=config.data.max_scramble
    )
    print("Dataset size:", len(ds_inmem))

    dl = DataLoader(
        ds_inmem,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2) Diffusion 模块与模型
    diffusion = DiscreteDiffusionLM(
        mask_id=MASK_TOKEN,  # 仅用于噪声
        schedule_start=config.diffusion.schedule_start,
        schedule_end=config.diffusion.schedule_end,
        num_steps=config.diffusion.num_steps
    )

    model = DiffusionLMTransformer(
        vocab_size=VOCAB_SIZE,  # 例如 22
        cond_dim=config.model.cond_dim,  # 例如 54
        d_model=config.model.d_model,  # 例如 128
        n_layers=config.model.n_layers,  # 例如 4
        n_heads=config.model.n_heads,  # 例如 4
        p_drop=config.model.p_drop,  # 例如 0.1
        max_seq_len=config.model.max_seq_len  # 例如 50
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    # CE时, ignore_index=PAD_TOKEN (例如18), 使得填充区不计入loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    epochs = config.training.epochs
    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        # 使用 tqdm 显示每个 epoch 的进度
        epoch_bar = tqdm(dl, desc=f"Epoch {ep}/{epochs}", leave=False)
        for conds, seqs, lengths in epoch_bar:
            conds = conds.to(device).float()  # (B, cond_dim)
            seqs = seqs.to(device)  # (B, seq_len)
            B, L = seqs.shape

            # 随机选取噪声步数，并生成被噪声干扰的输入
            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            x_t = diffusion.q_sample(seqs, t)  # 替换为 MASK_TOKEN

            # 前向传播
            logits = model(x_t, conds)  # (B, seq_len, VOCAB_SIZE)
            loss = criterion(logits.view(-1, VOCAB_SIZE), seqs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B
            global_step += 1

            # 在进度条中实时更新当前 loss
            epoch_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_count if total_count > 0 else 0.0
        print(f"Epoch {ep}, Loss={avg_loss:.4f}")
        experiment.log_metric("train_loss", avg_loss, step=ep)

    # 保存最终模型，并通过 Comet ML 记录模型
    torch.save(model.state_dict(), "diffusion_model_final.pth")
    log_model(experiment, model=model, model_name="DiffusionLMTransformer")
    print("Training completed. Model saved as diffusion_model_final.pth")


if __name__ == "__main__":
    train_diffusion_lm_condition()
