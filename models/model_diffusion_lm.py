import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from dataset_rubik_seq import RubikSeqDataset
from dataset_rubik_seq import collate_fn

# 从 util.py 引入修改后的常量 & 函数
from utils import (
    COLOR_CHARS,
    MOVES_POOL,
    MOVE_TO_IDX,
    PAD_TOKEN,
    MASK_OR_NOMOVE_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    VOCAB_SIZE,
    convert_state_to_tensor,
    move_str_to_idx,
)

#########################################
# 1) Dataset: (初始状态, 解法序列) -> (cond, [SOS] + moves + [EOS])
#########################################
class RubikDiffusionDataset(Dataset):
    """
    仅示例用: (cond(54), seq) => seq含 [SOS], [EOS].
    这里随机生成假数据, 真实场景应读取实际魔方数据.
    """
    def __init__(self, size=1000, max_seq_len=20):
        super().__init__()
        self.size = size
        self.max_seq_len = max_seq_len
        self.data = []
        for _ in range(size):
            # cond: shape=(54,), 随机模拟
            cond = torch.randint(0, len(COLOR_CHARS), (54,))  # 0..5
            # seq len
            seq_len = random.randint(3, max_seq_len - 2)
            # 随机选 moves
            moves = [random.choice(MOVES_POOL) for _ in range(seq_len)]
            seq_ids = [SOS_TOKEN] + [move_str_to_idx(m) for m in moves] + [EOS_TOKEN]
            self.data.append((cond, torch.tensor(seq_ids, dtype=torch.long)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


#########################################
# 2) DiscreteDiffusionLM
#    - 在前向扩散中, 以一定概率将 token 替换为 MASK_TOKEN=19
#########################################
class DiscreteDiffusionLM:
    def __init__(self,
                 mask_id=MASK_OR_NOMOVE_TOKEN,  # 改用MASK_TOKEN=19
                 schedule_start=0.1,
                 schedule_end=0.7,
                 num_steps=10):
        self.mask_id = mask_id
        self.num_steps = num_steps
        self.schedule_start = schedule_start
        self.schedule_end = schedule_end

    def mask_prob(self, t):
        """线性插值"""
        return self.schedule_start + (self.schedule_end - self.schedule_start)*(t/(self.num_steps-1))

    def q_sample(self, x0, t):
        """
        x0: (B,L)
        t: (B,)
        => x_t: (B,L)
        在 x0 中以mask_prob(t)概率替换成 mask_id(=19)
        """
        B, L = x0.shape
        x_t = x0.clone()
        for i in range(B):
            ti = t[i].item()
            p = self.mask_prob(ti)
            # 逐元素随机
            mask_flag = torch.rand(L, device=x0.device) < p
            x_t[i][mask_flag] = self.mask_id
        return x_t


#########################################
# 3) 模型: DiffusionLMTransformer (带 condition)
#    vocab_size=22 (0..17动作,18=PAD,19=MASK,20=EOS,21=SOS)
#########################################
class DiffusionLMTransformer(nn.Module):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,  # 22
                 cond_dim=54,
                 d_model=128,
                 n_layers=4,
                 n_heads=4,
                 p_drop=0.1,
                 max_seq_len=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.d_model = d_model

        # token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # position embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # condition -> embedding
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=p_drop,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x_t, cond):
        """
        x_t: (B,L) 离散token序列, 包括 [MASK],[PAD],[EOS],[SOS]...
        cond: (B,54)
        => logits: (B,L,vocab_size)
        """
        B, L = x_t.shape

        # token & pos embedding
        tok_emb = self.token_emb(x_t)  # (B,L,d_model)
        pos_idx = torch.arange(L, device=x_t.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_idx)  # (1,L,d_model)
        x = tok_emb + pos_emb  # (B,L,d_model)

        # cond_emb
        cond_emb = self.cond_proj(cond)  # (B,d_model)
        x = x + cond_emb.unsqueeze(1)    # broadcast to each position

        h = self.transformer(x)  # (B,L,d_model)
        h = self.ln(h)
        logits = self.head(h)
        return logits


#########################################
# 4) 训练流程
#########################################
def train_diffusion_lm_condition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset
    ds_inmem = RubikSeqDataset(num_samples=10000, min_scramble=8, max_scramble=25)
    print("Dataset size:", len(ds_inmem))

    dl = DataLoader(ds_inmem, batch_size=128, shuffle=True, collate_fn=collate_fn)

    # 2) Diffusion + 模型
    diffusion = DiscreteDiffusionLM(
        mask_id=MASK_OR_NOMOVE_TOKEN,        # 仅用于噪声
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
