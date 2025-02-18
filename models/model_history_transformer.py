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

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor, shape (B, src_seq_len, input_dim)
            tgt: Tensor, shape (B, tgt_seq_len)，包含 move 的索引

        Returns:
            logits: Tensor, shape (B, tgt_seq_len, num_moves)
        """
        B, src_seq_len, _ = src.shape
        B, tgt_seq_len = tgt.shape

        # 处理 Encoder 输入
        # 1. 将输入维度调整为 (src_seq_len, B, input_dim)
        src = src.permute(1, 0, 2)
        src = src.float()
        # 2. 线性映射到 d_model
        src = self.src_linear(src)
        # 3. 添加位置编码
        # 这一段代码什么意思，我没太弄懂，src_positions为什么要弄成shape为(src_seq_len, 1)
        # 实际上src_seq_len只是复原到魔方的相对位置，不是绝对位置
        # 比如在第一条数据中1可能是全局数据中的第10条，而第二条数据中1可能是全局数据中的第5条
        src_positions = torch.arange(src_seq_len, device=src.device).unsqueeze(1)  # (src_seq_len, 1)
        src = src + self.src_pos_embedding(src_positions)  # (src_seq_len, B, d_model)

        # 处理 Decoder 输入
        tgt_input = tgt[:, :-1]  # 例如 [SOS, move1, move2, ..., move_{n-1}]
        tgt_input = tgt_input.permute(1, 0)  # (tgt_seq_len-1, B)
        tgt_emb = self.tgt_embedding(tgt_input)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)
        tgt_emb = tgt_emb + self.tgt_pos_embedding(tgt_positions)

        # 生成 decoder 的因果掩码，长度为 tgt_emb 的时间步数 (即 tgt_seq_len-1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)

        # 调用 Transformer 模型，得到 decoder 的输出
        # 注意：这里 src_mask、memory_mask、以及 padding mask 可以根据需要进一步补充
        out = self.transformer(src=src, tgt=tgt_emb, tgt_mask=tgt_mask)
        # out: (tgt_seq_len, B, d_model)

        # 将输出转换回 (B, tgt_seq_len, d_model)
        out = out.permute(1, 0, 2)
        # 投影到 move 词汇表上
        logits = self.fc_out(out)  # (B, tgt_seq_len, num_moves)
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
