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

import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_no_teacher_forcing(model, diffusion, val_dl, device):
    """
    在验证集上做 "多步后向采样" (不使用Teacher Forcing)，
    并计算最终的 token-level Accuracy。

    假设:
      - model: DiffusionLMTransformer
      - diffusion: DiscreteDiffusionLM (含 num_steps, etc.)
      - val_dl: 验证集 DataLoader，每个batch给出 (conds, seqs, lengths)
      - 我们只在结尾对 "最终生成序列" 与 "真实序列" 做一次对比, 忽略 [PAD] 位置.

    返回: final_accuracy (float)
    """

    model.eval()
    total_correct = 0
    total_nonpad = 0

    for conds, seqs, lengths in val_dl:
        # conds: (B, cond_dim)
        # seqs:  (B, seq_len), 包含动作token, 以及 [PAD] (18), [MASK](19?), [EOS](20), [SOS](21)等
        conds = conds.to(device).float()
        seqs = seqs.to(device)  # gold sequence (B,L), 用于对比

        B, L = seqs.shape
        # 1) 初始化 x_T: 除了 [PAD] 位置外, 其它全部设为 [MASK]
        #    因为在真正"无teacher forcing"多步采样时, 仅保留PAD原样(它只是占位)
        x_t = seqs.clone()
        # mask_flag: (B,L) = (x_t != PAD_TOKEN)
        mask_flag = (x_t != diffusion.pad_id) if hasattr(diffusion,"pad_id") else (x_t != 18)
        # 把非PAD位置都变成 [MASK_TOKEN]
        x_t[mask_flag] = diffusion.mask_id  # e.g. 19

        # 2) 多步逆扩散: 从 t = num_steps-1 down to 0
        T = diffusion.num_steps
        for step in reversed(range(T)):
            # logits: (B,L,vocab_size)
            # 这里假设你的 model.forward 接口 => model(x_t, conds)
            # 若模型需要 time step输入( e.g. model(x_t, conds, step) )，请自行改动
            logits = model(x_t, conds)
            # 得到分类概率
            prob = F.softmax(logits, dim=-1)

            # 仅对当前 [MASK] 的位置进行 argmax
            still_masked = (x_t == diffusion.mask_id)  # (B,L) bool
            if still_masked.any():
                pred_tokens = torch.argmax(prob, dim=-1)  # (B,L)
                # 把 still_masked 的位置替换成 pred_tokens
                x_t[still_masked] = pred_tokens[still_masked]

            # 这里的示例是一种简化——"一步填满所有MASK"
            # 更细致的做法: 可能只替换部分token, 或结合置信度等策略

        # 3) 现在 x_t 已是 "模型最终生成" 的序列 (B,L)
        #    对比 seqs(真实) => 忽略 [PAD]
        # token-level
        not_pad = (seqs != diffusion.pad_id) if hasattr(diffusion,"pad_id") else (seqs != 18)
        correct = (x_t == seqs) & not_pad
        total_correct += correct.sum().item()
        total_nonpad += not_pad.sum().item()

    final_acc = (total_correct / total_nonpad) if total_nonpad>0 else 0.0
    return final_acc

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
        max_scramble=config.data.max_scramble,
        use_redis=True
    )
    print("Dataset size:", len(ds_inmem))
    val_ds_inmem = RubikSeqDataset(
        num_samples=1000,
        min_scramble=config.data.min_scramble,
        max_scramble=config.data.max_scramble,
        use_redis=True,
        redis_db=1
    )
    dl = DataLoader(
        ds_inmem,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds_inmem,
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
        # === 验证: 无teacher forcing, 多步采样准确率
        val_acc = evaluate_no_teacher_forcing(model, diffusion, val_dl, device)

        print(f"Epoch {ep} | train_loss={avg_loss:.4f} | no_TF_val_acc={val_acc:.4f}")

        # comet ml记录
        experiment.log_metric("train_loss", avg_loss, step=ep)
        experiment.log_metric("val_accuracy", val_acc, step=ep)

    # 保存最终模型，并通过 Comet ML 记录模型
    torch.save(model.state_dict(), "diffusion_model_final.pth")
    log_model(experiment, model=model, model_name="DiffusionLMTransformer")
    print("Training completed. Model saved as diffusion_model_final.pth")


if __name__ == "__main__":
    train_diffusion_lm_condition()
