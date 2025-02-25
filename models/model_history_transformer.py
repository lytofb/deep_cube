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
                 max_seq_len=50,
                 dropout = 0.3,
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

        # 新增：每个面的位置嵌入（6个面，位置向量维度设为 pos_dim，例如16）
        self.pos_dim = 16
        self.face_position_embedding = nn.Embedding(6, self.pos_dim)
        self.face_pos_linear = nn.Linear(self.pos_dim, d_model)  # 投影到 d_model 维度

        # 新增：用于将每个面原始 9 维特征转换为 d_model 维（方案一使用）
        self.face_feature_linear = nn.Linear(9, d_model)

        # 新增：卷积层用于捕获局部结构（利用一个合理的魔方展开图，例如十字型布局）
        self.cube_net_conv = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.cube_net_activation = nn.ReLU()
        self.cube_net_dropout = nn.Dropout(dropout)

        # 1) 在输入 Embedding 上增加 Dropout
        self.src_emb_dropout = nn.Dropout(dropout)
        self.tgt_emb_dropout = nn.Dropout(dropout)

        # 2) 对 Encoder/Decoder 的输出增加 Dropout（原有的 dropout1 也可保留）
        self.dropout1 = nn.Dropout(dropout)


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
            dim_feedforward=d_model * 4,
            dropout=dropout  # <-- 让 Transformer 自身的多头注意力和前馈层也应用 Dropout
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
        # ① 提取魔方状态：假定前54维代表6个面（每面9个数）
        cube_state = src[..., :54].view(B, src_seq_len, 6, 9)  # (B, T, 6, 9)

        # ② 计算每个面的特征向量（不考虑空间布局，此处每面为一个向量）
        face_features = self.face_feature_linear(cube_state)  # (B, T, 6, d_model)

        # ③ 获取每个面的位置信息（固定面索引 0~5）
        face_ids = torch.arange(6, device=src.device).unsqueeze(0).unsqueeze(0)  # (1,1,6)
        face_pos_embed = self.face_position_embedding(face_ids)  # (1,1,6, pos_dim)
        face_pos_embed_proj = self.face_pos_linear(face_pos_embed)  # (1,1,6, d_model)

        # ④ 融合位置信息：加法融合（广播后）
        face_features = face_features + face_pos_embed_proj  # (B, T, 6, d_model)

        # ⑤ 为了捕获局部空间结构，使用卷积处理每个面原始的 3×3 布局：
        cube_faces = cube_state.view(B, src_seq_len, 6, 3, 3).float()  # (B, T, 6, 3, 3)
        # 对每个面单独处理：合并 batch & time & face 维度
        faces_reshaped = cube_faces.view(B * src_seq_len * 6, 1, 3, 3)  # (B*T*6, 1, 3, 3)
        conv_out = self.cube_net_conv(faces_reshaped)  # (B*T*6, d_model, 3, 3)
        conv_out = self.cube_net_activation(conv_out)
        conv_out = self.cube_net_dropout(conv_out)
        # 全局池化，得到每个面的卷积特征向量
        conv_face_feat = conv_out.view(B * src_seq_len * 6, self.d_model, -1).mean(dim=2)  # (B*T*6, d_model)
        conv_face_feat = conv_face_feat.view(B, src_seq_len, 6, self.d_model)  # (B, T, 6, d_model)

        # ⑥ 融合：例如对比两种特征，可取平均（也可拼接后投影）
        fused_face_feat = (face_features + conv_face_feat) / 2  # (B, T, 6, d_model)

        # ⑦ 聚合6个面的信息（例如求平均），作为每个时间步的编码表示
        cube_enhanced = fused_face_feat.mean(dim=2)  # (B, T, d_model)

        # 后续：使用 cube_enhanced 替代原先经过 self.src_linear 得到的表示
        src = cube_enhanced.permute(1, 0, 2)  # (T, B, d_model)
        src_positions = torch.arange(src_seq_len, device=src.device).unsqueeze(1)
        src = src + self.src_pos_embedding(src_positions)
        src = self.src_emb_dropout(src)

        # ------- Decoder Embedding -------
        tgt_input = tgt_input.permute(1, 0)  # => (tgt_seq_len-1, B)
        tgt_emb = self.tgt_embedding(tgt_input)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)
        tgt_emb = tgt_emb + self.tgt_pos_embedding(tgt_positions)

        # 在 Decoder 输入阶段也加个 Dropout
        tgt_emb = self.tgt_emb_dropout(tgt_emb)

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
