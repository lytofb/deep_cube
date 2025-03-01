import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# ---------------------------
# 1. 定义配置类
# ---------------------------
class RubikSeq2SeqConfig(PretrainedConfig):
    model_type = "rubik-seq2seq"
    def __init__(self,
                 input_dim=55,       # 每个时间步的特征维度（例如魔方状态信息，如55维）
                 d_model=128,        # Transformer 内部特征维度
                 nhead=4,            # 多头注意力头数
                 num_layers=12,      # Encoder 和 Decoder 的层数
                 num_moves=21,       # move 的总种类数（词汇大小）
                 max_seq_len=50,     # 序列最大长度（用于位置编码）
                 dropout=0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_moves = num_moves
        self.max_seq_len = max_seq_len
        self.dropout = dropout

# ---------------------------
# 2. 定义转换后的模型类
# ---------------------------
class RubikSeq2SeqForConditionalGeneration(PreTrainedModel):
    config_class = RubikSeq2SeqConfig

    def __init__(self, config: RubikSeq2SeqConfig):
        super().__init__(config)
        self.config = config

        # Encoder 部分：对魔方状态进行线性映射 + 位置编码
        self.src_linear = nn.Linear(config.input_dim, config.d_model)
        self.src_pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.src_emb_dropout = nn.Dropout(config.dropout)

        # Decoder 部分：对 move 索引进行嵌入 + 位置编码
        self.tgt_embedding = nn.Embedding(config.num_moves, config.d_model, padding_idx=19)
        self.tgt_pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.tgt_emb_dropout = nn.Dropout(config.dropout)

        # Encoder-Decoder 之间和 Transformer 后的 Dropout
        self.dropout1 = nn.Dropout(config.dropout)

        # Transformer 模型（同时包含 Encoder 和 Decoder）
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout
        )

        # 输出层：将 Transformer 的输出映射到 move 的词汇表上
        self.fc_out = nn.Linear(config.d_model, config.num_moves)

        # 初始化权重（调用 Hugging Face 内部的方法）
        self.init_weights()

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成 tgt 的因果掩码，构造一个上三角矩阵，使得解码器在预测时无法看到未来的信息
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                input_ids: torch.Tensor,          # encoder 输入：形状 (B, src_seq_len, input_dim)
                decoder_input_ids: torch.Tensor,    # decoder 输入：形状 (B, tgt_seq_len)
                return_hidden: bool = False,        # 是否返回隐藏状态（用于 PPO 中的 critic 计算）
                **kwargs):
        """
        Args:
            input_ids: 编码器输入，包含魔方状态信息，形状 (B, src_seq_len, input_dim)
            decoder_input_ids: 解码器输入（move 索引），形状 (B, tgt_seq_len)
            return_hidden: 如果为 True，则返回 (hidden, logits)
        Returns:
            如果 return_hidden 为 False，则返回 logits，形状 (B, tgt_seq_len, num_moves)
            如果 return_hidden 为 True，则返回 (hidden, logits)：
                - hidden 的形状为 (B, tgt_seq_len, d_model)
                - logits 的形状为 (B, tgt_seq_len, num_moves)
        """
        B, src_seq_len, _ = input_ids.shape
        B2, tgt_seq_len = decoder_input_ids.shape
        assert B == B2, "Encoder 和 Decoder 的 batch size 必须一致。"

        # ----- Encoder 部分 -----
        # 将输入转置为 (src_seq_len, B, input_dim) 后进行线性映射，再加上位置编码
        src = input_ids.permute(1, 0, 2).float()         # (src_seq_len, B, input_dim)
        src = self.src_linear(src)                         # (src_seq_len, B, d_model)
        src_positions = torch.arange(src_seq_len, device=input_ids.device).unsqueeze(1)  # (src_seq_len, 1)
        src = src + self.src_pos_embedding(src_positions)
        src = self.src_emb_dropout(src)

        # ----- Decoder 部分 -----
        # 解码器输入为 token 索引，先做嵌入，再加上位置编码
        tgt = decoder_input_ids.transpose(0, 1)            # (tgt_seq_len, B)
        tgt_emb = self.tgt_embedding(tgt)                  # (tgt_seq_len, B, d_model)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)  # (tgt_seq_len, 1)
        tgt_emb = tgt_emb + self.tgt_pos_embedding(tgt_positions)
        tgt_emb = self.tgt_emb_dropout(tgt_emb)

        # 生成解码器的因果掩码，保证自回归预测时只依赖历史信息
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)

        # ----- Transformer 前向传播 -----
        out = self.transformer(
            src=src,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
        )
        out = out.transpose(0, 1)   # (B, tgt_seq_len, d_model)
        hidden = self.dropout1(out)
        logits = self.fc_out(hidden)    # (B, tgt_seq_len, num_moves)
        if return_hidden:
            return hidden, logits
        return logits

    def prepare_inputs_for_generation(self, decoder_input_ids, **kwargs):
        # 在使用 Hugging Face 的 generate() 方法时，返回解码器输入即可
        return {"decoder_input_ids": decoder_input_ids}


# ---------------------------
# 3. 测试转换后的模型并加载预训练权重
# ---------------------------
if __name__ == "__main__":
    # 根据 config.yaml 中的模型配置初始化配置对象
    config = RubikSeq2SeqConfig(
        input_dim=55,
        d_model=256,
        nhead=8,
        num_layers=6,
        num_moves=21,
        max_seq_len=50,
        dropout=0.2
    )
    # 根据配置初始化模型
    model = RubikSeq2SeqForConditionalGeneration(config)

    # 加载之前保存的权重文件 rubik_model_best.pth
    state_dict = torch.load("rubik_model_best.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    print("成功加载 rubik_model_best.pth 权重。")

    # 假设 batch_size = 2，src 序列长度为 8，tgt 序列长度为 8
    B = 2
    src_seq_len = 8
    tgt_seq_len = 8

    # encoder 输入：随机生成魔方状态（浮点数），形状 (B, src_seq_len, input_dim)
    src = torch.randn(B, src_seq_len, config.input_dim)
    # decoder 输入：随机生成 move 序列（索引），形状 (B, tgt_seq_len)
    tgt = torch.randint(0, config.num_moves, (B, tgt_seq_len))

    # 测试 forward：默认返回 logits
    logits = model(src, tgt)
    print("logits shape:", logits.shape)  # 预期输出: (B, tgt_seq_len, num_moves)

    # 测试 forward：返回 (hidden, logits)
    hidden, logits = model(src, tgt, return_hidden=True)
    print("hidden shape:", hidden.shape)  # (B, tgt_seq_len, d_model)
    print("logits shape:", logits.shape)
