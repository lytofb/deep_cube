import torch
import torch.nn as nn

from models.positional_embedding import SinusoidalPosEmb
from utils import PAD_TOKEN,VOCAB_SIZE
from typing import Union, Optional, Tuple

import math, types
import torch.nn.functional as F

############################################################
# <<< NEW: VQ EMBEDDING MODULE >>>
############################################################
class VQEmbedding(nn.Module):
    """Vector‑Quantised Embedding layer (EMA‑style, codebook learned).

    Args:
        num_embeddings: size of the codebook.
        embedding_dim: dimensionality of each embedding vector.
        commitment_cost: beta coefficient described in the VQ‑VAE paper.
        decay: decay rate for EMA updates.
        eps: small value to avoid divide‑by‑zero.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook (K, D)
        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_emb_sum", torch.randn(num_embeddings, embedding_dim))

        nn.init.uniform_(self.embeddings, -1.0 / num_embeddings, 1.0 / num_embeddings)
        nn.init.uniform_(self.ema_emb_sum, -1.0 / num_embeddings, 1.0 / num_embeddings)

    @torch.no_grad()
    def _ema_update(self, flat_inputs, encoding_indices):
        # one‑hot encode indices
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)
        cluster_size = encodings.sum(dim=0)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        emb_sum = encodings.t() @ flat_inputs
        self.ema_emb_sum.mul_(self.decay).add_(emb_sum, alpha=1 - self.decay)

        # Normalize to get updated embeddings
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
        self.embeddings.copy_(self.ema_emb_sum / cluster_size.unsqueeze(1))

    def forward(self, x: torch.Tensor):
        """Args:
            x: Tensor of shape (..., D)
        Returns:
            quantized: same shape as x
            vq_loss: commitment + codebook losses (scalar)
            indices: codebook indices (flattened view)
        """
        shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)

        # Compute distances (||x||^2 - 2 x·e + ||e||^2)
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embeddings[indices].view(*shape)

        # Commitment & codebook loss
        vq_loss = (
            F.mse_loss(quantized.detach(), x) +
            self.commitment_cost * F.mse_loss(quantized, x.detach())
        )

        # EMA update (training mode only)
        if self.training:
            self._ema_update(flat_x, indices)

        # Straight‑through estimator
        quantized = x + (quantized - x).detach()
        return quantized, vq_loss, indices.view(shape[:-1])


class SrcEmbeddingSeparate(nn.Module):
    def __init__(self, vocab_size, d_model, input_dim):
        """
        参数：
          - vocab_size: 词表大小，用于nn.Embedding的输入
          - d_model: 每个 token 的嵌入维度
          - input_dim: 每个输入 token 的字段数（例如前 input_dim-1 个字段和最后 1 个字段分开激活）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        # 对于前 input_dim-1 部分（连续拼接后维度为 (input_dim-1)*d_model）进行投影和激活
        self.project_first = nn.Sequential(
            nn.Linear(d_model * (input_dim - 1), d_model),
            nn.Mish(),  # 使用Mish激活，当然也可换成其他激活函数
        )
        # 对于最后1个字段（维度 d_model）采用单独的投影+激活
        self.project_last = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),  # 同样也可选择其他激活函数
        )
        # 融合两部分特征，再投影回 d_model 输出维度
        self.fuse = nn.Linear(d_model * 2, d_model)

    def forward(self, src):
        """
        src.shape == (B, seq, input_dim)
        先进行embedding后得到 (B, seq, input_dim, d_model)
        """
        # (B, seq, input_dim, d_model)
        x = self.embedding(src)
        B, seq, in_dim, d_model_ = x.shape

        # 按字段拆分：假设前 in_dim-1 和最后1字段分开处理
        x_first = x[:, :, :in_dim - 1, :]   # (B, seq, in_dim-1, d_model)
        x_last  = x[:, :, in_dim - 1:, :]     # (B, seq, 1, d_model)

        # 将各部分的最后两个维度展平
        # 前部分：将 (in_dim-1, d_model) flatten 成 ( (in_dim-1)*d_model )
        x_first_flat = x_first.view(B, seq, (in_dim - 1) * d_model_)
        # 后部分：将 (1, d_model) flatten 成 (d_model)
        x_last_flat = x_last.view(B, seq, d_model_)

        # 分别经过独立投影和激活
        first_out = self.project_first(x_first_flat)  # (B, seq, d_model)
        last_out  = self.project_last(x_last_flat)      # (B, seq, d_model)

        # 连接：沿最后一维拼接 => (B, seq, 2*d_model)
        combined = torch.cat([first_out, last_out], dim=-1)
        # 融合得到最终输出： (B, seq, d_model)
        out = self.fuse(combined)
        return out

class SrcEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.project = nn.Sequential(
            nn.Linear(d_model * input_dim, d_model),
            nn.Mish(),
        )

    def forward(self, src):
        """
        src.shape == (B, seq, input_dim)
        """
        # (B, seq, input_dim, d_model)
        x = self.embedding(src)

        # 先把 input_dim 这一维合并到最后一个维度上 => (B, seq, input_dim*d_model)
        B, seq, in_dim, d_model_ = x.shape
        x = x.view(B, seq, in_dim * d_model_)

        # 再投影回 (B, seq, d_model)
        x = self.project(x)
        return x

class SrcLinearSimpleModel(nn.Module):
    def __init__(self,input_dim,d_model):
        super(SrcLinearSimpleModel, self).__init__()
        self.linear_activation = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        )

    def forward(self,src):
        src = src.float()
        out = self.linear_activation(src)
        return out

class SrcLinearModel(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcLinearModel, self).__init__()
        # 假设最终想要的输出维度是 d_model
        # 前 (input_dim-1) -> d_model/2
        self.linear_left = nn.Linear(input_dim - 1, d_model)
        # 最后 1 -> d_model/2
        self.linear_right = nn.Linear(1, d_model)
        self.activation = nn.Sequential(
            nn.Mish(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, src):
        # src 的形状: (B, src_seq_len, input_dim)
        # 取前面 (input_dim-1) 维:
        left_part = src[:, :, :-1].float()  # shape: (B, src_seq_len, input_dim-1)
        # 取最后 1 维，并保留其维度:
        right_part = src[:, :, -1:].clone().float()  # shape: (B, src_seq_len, 1)

        # 分别过线性映射
        left_out = self.linear_left(left_part)  # shape: (B, src_seq_len, d_model/2)
        right_out = self.linear_right(right_part)  # shape: (B, src_seq_len, d_model/2)

        # 在最后一个维度拼接
        out = torch.cat([left_out, right_out], dim=-1)  # shape: (B, src_seq_len, d_model)
        out = self.activation(out)
        return out

### <<< NEW / MODIFY >>>  (2)  LoRA 低秩线性层
class LoRALinear(nn.Module):
    """
    将已有 nn.Linear 替换为带可训练 LoRA 分支的线性层。
    公式:  y = Wx + (α/r)·B(Ax)  （W = 原权重，A↓r，B↑）
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0,
                 freeze_base: bool = True):
        super().__init__()
        self.base = base                          # 原线性层（冻结 or 不冻结按需控制）
        if freeze_base:                 # **可选：自动冻结基座**
            for p in self.base.parameters():
                p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_down = nn.Linear(base.in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.dropout = nn.Dropout(dropout)

    # 让外部能访问到 weight / bias
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_up(self.lora_down(self.dropout(x)))

### <<< NEW / MODIFY >>>  (3)  Bottleneck Adapter
class Adapter(nn.Module):
    """
    简单的 Bottleneck Adapter：x + Dropout(Up(GELU(Down(x))))
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.up(self.act(self.down(x))))


class RubikEncoderOnly(nn.Module):
    """
    该模型用于学习从魔方状态序列到还原 move 序列的映射。

    输入：
      - src: 魔方状态序列，形状 (B, src_seq_len, input_dim)，input_dim（例如55）中包含魔方状态信息（贴纸颜色等）。
      - tgt: move 序列（作为 decoder 的输入，教师强制时使用），形状 (B, tgt_seq_len)，每个元素为 move 的索引。

    输出：
      - logits: 预测每个时间步的 move 分布，形状 (B, tgt_seq_len, num_moves)
    """

    ### <<< NEW / MODIFY >>>  (6)  注入 LoRA 的递归函数
    def _inject_lora(self, module: nn.Module, r: int = 8, alpha: int = 16, freeze_base=True):
        """
        递归地把 module 里所有 nn.Linear 替换成 LoRALinear。
        """
        for name, child in list(module.named_children()):  # list() 防止迭代时修改
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, freeze_base=freeze_base))
            else:
                self._inject_lora(child, r, alpha, freeze_base)

    ### <<< NEW / MODIFY >>>  (7)  给 Encoder 每一层打 Adapter「补丁」
    def _add_adapters(self, bottleneck_dim: int = 32):
        """
        给 TransformerEncoder 每层打 Adapter 补丁，保持原 forward 签名。
        """
        for layer in self.encoder.layers:  # type: nn.TransformerEncoderLayer
            layer.adapter = Adapter(self.d_model, bottleneck_dim)
            old_forward = layer.forward

            def forward_with_adapter(self_layer, src, *args, **kwargs):
                out = old_forward(src, *args, **kwargs)
                return self_layer.adapter(out)

            layer.forward = forward_with_adapter.__get__(layer, layer.__class__)

    def __init__(self,
                 input_dim=55,
                 d_model=128,
                 nhead=4,
                 num_layers=6,
                 num_moves=VOCAB_SIZE,
                 max_seq_len=50,
                 dropout = 0.3,
                 # ---- 新增 ----
                 use_lora: bool = False,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 use_adapter: bool = False,
                 adapter_dim: int = 32,
                 # VQ Embedding
                 use_vq: bool = False,
                 vq_codebook_size: int = 512,
                 vq_commitment_cost: float = 0.25
                 ):
        """
        Args:
            input_dim: 每个时间步的特征维度（例如魔方状态特征，如54贴纸+1 move信息）
            d_model: Transformer 内部特征维度
            nhead: 多头注意力的头数
            num_layers: Encoder 和 Decoder 层数
            num_moves: move 的总种类数（词汇大小）
            max_seq_len: 序列的最大长度，用于位置编码
        """
        super().__init__()
        self.use_lora     = use_lora
        self.lora_r       = lora_r
        self.lora_alpha   = lora_alpha
        self.use_adapter  = use_adapter
        self.adapter_dim  = adapter_dim

        self.use_vq = use_vq

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_moves = num_moves
        self.max_seq_len = max_seq_len

        # VQ module (optional)
        if self.use_vq:
            self.vq = VQEmbedding(vq_codebook_size, d_model, commitment_cost=vq_commitment_cost)
        else:
            self.vq = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # --- NEW: Prompt Embedding, 2 个索引 {0: 普通, 1: 首步} ---
        self.prompt_embedding = nn.Embedding(2, d_model)

        self.prompt_alpha = nn.Parameter(torch.tensor(0.0))
        self.pos_alpha = nn.Parameter(torch.tensor(1.0))

        # __init__
        # self.prompt_film = nn.Linear(d_model, 2 * d_model)  # 生成 γ,β

        # 1) 在输入 Embedding 上增加 Dropout
        self.src_emb_dropout = nn.Dropout(dropout)


        # Encoder：对魔方状态进行线性映射，然后加上位置编码
        self.src_embedding = SrcEmbeddingSeparate(num_moves, d_model,input_dim)
        # self.src_embedding = SrcLinearModel(input_dim, d_model)
        # self.src_embedding = SrcLinearSimpleModel(input_dim, d_model)
        self.src_pos_embedding = SinusoidalPosEmb(d_model)
        # self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # ---------- 注入 Bottleneck Adapter ----------
        if self.use_adapter:
            self._add_adapters(self.adapter_dim)

        # ---------- 注入 LoRA ----------
        if self.use_lora:
            self._inject_lora(self.encoder, r=self.lora_r, alpha=self.lora_alpha)
            # 如还想在输出 MLP 里也用 LoRA，可解开下面一行
            # self._inject_lora(self.fc_out, r=self.lora_r, alpha=self.lora_alpha)



        # self.encoder = nn.Sequential(
        #     nn.Linear(d_model, 4 * d_model),
        #     nn.Mish(),
        #     nn.Linear(4 * d_model, d_model)
        # )

        # decoder
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=4 * d_model,
        #     dropout=dropout,
        #     activation='gelu',
        #     batch_first=False,
        #     norm_first=True  # important for stability
        # )
        # self.decoder = nn.TransformerDecoder(
        #     decoder_layer=decoder_layer,
        #     num_layers=num_layers
        # )

        # Transformer 模型（包含 Encoder 和 Decoder）
        # self.transformer = nn.Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers,
        #     dim_feedforward=d_model * 4,
        #     dropout=dropout  # <-- 让 Transformer 自身的多头注意力和前馈层也应用 Dropout
        # )

        self.ln_f = nn.LayerNorm(d_model)
        # 输出层：将 Transformer 输出投影到 move 词汇表上
        # self.fc_out = nn.Linear(d_model, num_moves)

        # first_head：二层带激活的小 MLP
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_moves)
        )

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm,
                                    torch.nn.Embedding,
                                    Adapter,  # NEW
                                    LoRALinear)  # NEW
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("prompt_film")
        no_decay.add("cls_token")
        no_decay.add("prompt_alpha")
        no_decay.add("pos_alpha")
        # no_decay.add("_dummy_variable")
        # if self.cond_pos_emb is not None:
        #     no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def configure_lora_optimizers(
            self,
            learning_rate: float = 1e-3,  # LoRA 通常学习率更高
            weight_decay: float = 1e-2,
            betas: Tuple[float, float] = (0.9, 0.95)):
        """
        只优化 requires_grad=True 的参数。
        - LoRA / bias 不做权重衰减
        - 其余可选 weight_decay
        """

        decay, no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # 冻结的层直接跳过
            # LoRA 层或 bias / LayerNorm 等 -> no_decay
            if "lora_" in name or name.endswith("bias") or param.ndim == 1:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


    def generate_square_subsequent_mask(self, sz):
        """
        生成 tgt 的因果掩码，防止 decoder 看到未来信息
        """
        # 这个mask为什么要生成一个上三角矩阵，详细解释一下
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        Args:
            src:       shape (B, src_seq_len, input_dim)
        """
        B, src_seq_len, _ = src.shape

        # =========== 构建 Key Padding Mask ===========
        # 如果你的设计里, src[..., -1] 存放的是 token 索引，则下面这样判断
        # 否则要根据你的实际数据格式改写
        src_tokens = src[..., -1].long()           # (B, src_seq_len)
        src_key_padding_mask = (src_tokens == PAD_TOKEN)  # True 表示 padding，需要屏蔽

        # ------- Encoder 部分保持不变 -------
        src = src.permute(1, 0, 2).long()  # => (src_seq_len, B, d_model)
        src = self.src_embedding(src)
        # ---------- 插入 CLS token ----------
        cls_tok = self.cls_token.expand(1, B, -1)            # (1, B, d_model)
        src = torch.cat([cls_tok, src], dim=0)               # (L+1, B, d_model)
        src_positions = torch.arange(src.shape[0], device=src.device).unsqueeze(1)
        pos_emb = self.src_pos_embedding(src_positions)
        src = src + self.pos_alpha * pos_emb


        # ---------- 扩展 padding mask：给 CLS 位置补 False ----------
        cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=src.device)    # (B, 1)
        src_key_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)  # (B, L+1)

        # --- NEW: 计算首个非 PAD 的位置标记（包括 CLS） ---
        # not_pad: (B, L+1)，首个非 PAD（或 CLS）对应的位置是 1，其它 0
        not_pad = (~src_key_padding_mask).int()  # 1 表示真实 token
        first_flag = (not_pad.cumsum(dim=1) == 2).long()  # (B, L+1)
        # 转成 (L+1, B) 供 Embedding lookup
        prompt_ids = first_flag.transpose(0, 1)  # (L+1, B)
        prompt_emb = self.prompt_embedding(prompt_ids)  # (L+1, B, d_model)

        # forward 中，先算出 prompt_emb as before (L+1, B, d_model)
        # 然后
        # film = self.prompt_film(prompt_emb)  # (L+1, B, 2*d_model)
        # gamma, beta = film.chunk(2, dim=-1)  # 各 (L+1, B, d_model)
        # src = src * (1 + gamma) + beta

        # 把 prompt embedding 加回 src
        src = src + self.prompt_alpha * prompt_emb

        # 在 Encoder 输入阶段也加个 Dropout
        src = self.src_emb_dropout(src)

        # ---- VQ EMBEDDING HERE ----
        if self.vq is not None:
            # Convert to (B, L+1, D) for VQ, then back to (L+1, B, D)
            src_b = src.permute(1, 0, 2)
            src_b, vq_loss, _ = self.vq(src_b)
            src = src_b.permute(1, 0, 2)
        else:
            vq_loss = torch.tensor(0.0, device=src.device)

        out = self.encoder(src,src_key_padding_mask = src_key_padding_mask)

        out = out.permute(1, 0, 2)  # => (B, src_seq_len, d_model)
        out = self.ln_f(out)
        h_cls = out[:, 0, :]                  # (B, d_model) —— CLS 位置

        # ---------- 预测下一步动作 ----------
        logits = self.fc_out(h_cls)           # (B, num_moves)
        return logits


# 示例调用（注意：数据生成部分需要根据实际情况提供 src 和 tgt）：
if __name__ == "__main__":
    B = 2
    src_seq_len = 8  # 状态序列长度
    tgt_seq_len = 8  # move 序列长度
    input_dim = 55
    num_moves = VOCAB_SIZE

    model = RubikEncoderOnly(input_dim=input_dim, num_moves=num_moves, use_lora = True, use_adapter=True, use_vq=True)
    # state_dict = model.state_dict()
    # 假设我们已选取了某层的 weight
    # weight_matrix = state_dict['decoder.layers.0.self_attn.in_proj_weight'].cpu().numpy()

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(weight_matrix, cmap='viridis')
    # plt.title("Transformer Query Weight Matrix")
    # plt.xlabel("输出维度")
    # plt.ylabel("输入维度")
    # plt.show()
    print(model)
    src = torch.randint(0,22,(B, src_seq_len, input_dim))

    logits = model(src)  # (B, tgt_seq_len, num_moves)
    print(logits.shape)
