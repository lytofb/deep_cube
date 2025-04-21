import torch
import torch.nn as nn

from models.positional_embedding import SinusoidalPosEmb
from utils import PAD_TOKEN, VOCAB_SIZE, MASK_OR_NOMOVE_TOKEN
from typing import Union, Optional, Tuple

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

class RubikShortHorizonSeq2SeqTransformer(nn.Module):
    """
    该模型用于学习从魔方状态序列到还原 move 序列的映射。

    输入：
      - src: 魔方状态序列，形状 (B, src_seq_len, input_dim)，input_dim（例如55）中包含魔方状态信息（贴纸颜色等）。
      - tgt: move 序列（作为 decoder 的输入，教师强制时使用），形状 (B, tgt_seq_len)，每个元素为 move 的索引。

    输出：
      - logits: 预测每个时间步的 move 分布，形状 (B, tgt_seq_len, num_moves)
    """

    def __init__(self,
                 input_dim=55,
                 d_model=128,
                 nhead=4,
                 num_layers=6,
                 num_moves=VOCAB_SIZE,
                 max_seq_len=50,
                 dropout = 0.3,
                 src_seq_len=5,
                 tgt_seq_len=5,
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
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_moves = num_moves
        self.max_seq_len = max_seq_len

        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        # 1) 在输入 Embedding 上增加 Dropout
        self.src_emb_dropout = nn.Dropout(dropout)
        self.tgt_emb_dropout = nn.Dropout(dropout)

        # 2) 对 Encoder/Decoder 的输出增加 Dropout（原有的 dropout1 也可保留）
        self.dropout1 = nn.Dropout(dropout)


        # Encoder：对魔方状态进行线性映射，然后加上位置编码
        # self.src_embedding = SrcEmbedding(num_moves, d_model,input_dim)
        # self.src_embedding = SrcLinearModel(input_dim, d_model)
        self.src_embedding = SrcLinearSimpleModel(input_dim, d_model)
        self.src_pos_embedding = SinusoidalPosEmb(d_model)
        # self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Decoder：对 move 索引进行嵌入，并加上位置编码
        self.tgt_embedding = nn.Embedding(num_moves, d_model, padding_idx=PAD_TOKEN)
        self.tgt_pos_embedding = SinusoidalPosEmb(d_model)
        # self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=4 * d_model,
        #     dropout=dropout,
        #     activation='gelu',
        #     batch_first=False,
        #     norm_first=True
        # )
        # self.encoder = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer,
        #     num_layers=num_layers
        # )

        self.encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.Mish(),
            nn.Linear(4 * d_model, d_model)
        )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

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
        self.fc_out = nn.Linear(d_model, num_moves)

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
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
        # no_decay.add("pos_emb")
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

    def generate_square_subsequent_mask(self, sz):
        """
        生成 tgt 的因果掩码，防止 decoder 看到未来信息
        """
        # 这个mask为什么要生成一个上三角矩阵，详细解释一下
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt_input):
        """
        Args:
            src:       shape (B, src_seq_len, input_dim)
            tgt_input: shape (B, tgt_seq_len-1)  # 训练循环外部已经截断好
        """
        src = src[:,self.src_seq_len*(-1):,:]
        tgt_input = tgt_input[:,:self.tgt_seq_len]
        B, src_seq_len, _ = src.shape
        B, tgt_seq_len_minus1 = tgt_input.shape

        # =========== 构建 Key Padding Mask ===========
        # 如果你的设计里, src[..., -1] 存放的是 token 索引，则下面这样判断
        # 否则要根据你的实际数据格式改写
        src_tokens = src[..., -1].long()           # (B, src_seq_len)
        src_tokens[:, 0] = MASK_OR_NOMOVE_TOKEN
        src_key_padding_mask = (src_tokens == PAD_TOKEN)  # True 表示 padding，需要屏蔽
        tgt_key_padding_mask = (tgt_input == PAD_TOKEN)

        # ------- Encoder 部分保持不变 -------
        src = src.permute(1, 0, 2).long()  # => (src_seq_len, B, d_model)
        src = self.src_embedding(src)
        src_positions = torch.arange(src_seq_len, device=src.device).unsqueeze(1)
        src = src + self.src_pos_embedding(src_positions)

        # 在 Encoder 输入阶段也加个 Dropout
        src = self.src_emb_dropout(src)
        memory = self.encoder(src)

        # ------- Decoder Embedding -------
        tgt_input = tgt_input.permute(1, 0)  # => (tgt_seq_len-1, B)
        tgt_emb = self.tgt_embedding(tgt_input)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)
        tgt_emb = tgt_emb + self.tgt_pos_embedding(tgt_positions)

        # 在 Decoder 输入阶段也加个 Dropout
        tgt_emb = self.tgt_emb_dropout(tgt_emb)


        # ------- Causal Mask -------
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # ------- Transformer -------
        # out = self.transformer(
        #     src=src,
        #     tgt=tgt_emb,
        #     tgt_mask=tgt_mask,
        #     src_key_padding_mask=src_key_padding_mask,  # 屏蔽Encoder端PAD
        #     tgt_key_padding_mask=tgt_key_padding_mask,
        #     tgt_is_causal=True
        # )
        out = out.permute(1, 0, 2)  # => (B, tgt_seq_len-1, d_model)
        # out = self.dropout1(out)
        out = self.ln_f(out)
        logits = self.fc_out(out)  # => (B, tgt_seq_len-1, num_moves)
        return logits


# 示例调用（注意：数据生成部分需要根据实际情况提供 src 和 tgt）：
if __name__ == "__main__":
    B = 2
    src_seq_len = 8  # 状态序列长度
    tgt_seq_len = 8  # move 序列长度
    input_dim = 55
    num_moves = VOCAB_SIZE

    model = RubikShortHorizonSeq2SeqTransformer(input_dim=input_dim, num_moves=num_moves)
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

    src = torch.randint(0,22,(B, src_seq_len, input_dim))
    # src = torch.randn(B, src_seq_len, input_dim)  # 假设的魔方状态输入
    tgt = torch.randint(0, num_moves, (B, tgt_seq_len))  # 假设的 move 序列（索引）

    logits = model(src, tgt)  # (B, tgt_seq_len, num_moves)
    print(logits.shape)
