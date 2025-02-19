import torch
import torch.nn as nn

from utils import PAD_TOKEN


class RubikSeq2SeqTransformer(nn.Module):
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
                 num_layers=12,
                 num_moves=21,
                 max_seq_len=50):
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

        # Dropout layer
        self.dropout1 = nn.Dropout(p=0.2)

        # Encoder：对魔方状态进行线性映射，然后加上位置编码
        self.src_linear = nn.Linear(input_dim, d_model)
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Decoder：对 move 索引进行嵌入，并加上位置编码
        self.tgt_embedding = nn.Embedding(num_moves, d_model, padding_idx=19)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 模型（包含 Encoder 和 Decoder）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4
        )

        # 输出层：将 Transformer 输出投影到 move 词汇表上
        self.fc_out = nn.Linear(d_model, num_moves)

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
        B, src_seq_len, _ = src.shape
        B, tgt_seq_len_minus1 = tgt_input.shape

        # ------- Encoder 部分保持不变 -------
        src = src.permute(1, 0, 2).float()  # => (src_seq_len, B, d_model)
        src = self.src_linear(src)
        src_positions = torch.arange(src_seq_len, device=src.device).unsqueeze(1)
        src = src + self.src_pos_embedding(src_positions)

        # ------- Decoder Embedding -------
        tgt_input = tgt_input.permute(1, 0)  # => (tgt_seq_len-1, B)
        tgt_emb = self.tgt_embedding(tgt_input)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)
        tgt_emb = tgt_emb + self.tgt_pos_embedding(tgt_positions)

        # ------- Causal Mask -------
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)

        # ------- Transformer -------
        out = self.transformer(
            src=src,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            # src_key_padding_mask=...,   # 可选
            # tgt_key_padding_mask=...    # 可选
        )
        out = out.permute(1, 0, 2)  # => (B, tgt_seq_len-1, d_model)
        out = self.dropout1(out)
        logits = self.fc_out(out)  # => (B, tgt_seq_len-1, num_moves)
        return logits


# 示例调用（注意：数据生成部分需要根据实际情况提供 src 和 tgt）：
if __name__ == "__main__":
    B = 2
    src_seq_len = 8  # 状态序列长度
    tgt_seq_len = 8  # move 序列长度
    input_dim = 55
    num_moves = 18

    model = RubikSeq2SeqTransformer(input_dim=input_dim, num_moves=num_moves)
    src = torch.randn(B, src_seq_len, input_dim)  # 假设的魔方状态输入
    tgt = torch.randint(0, num_moves, (B, tgt_seq_len))  # 假设的 move 序列（索引）

    logits = model(src, tgt)  # (B, tgt_seq_len, num_moves)
    print(logits.shape)
