import torch
import torch.nn as nn

class RubikSeq2SeqTransformer(nn.Module):
    def __init__(self,
                 input_dim=55,
                 d_model=128,
                 nhead=4,
                 num_layers=12,
                 num_moves=21,
                 max_seq_len=50,
                 dropout=0.3,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_moves = num_moves
        self.max_seq_len = max_seq_len

        # face_feature_linear: 将每个面原始 9 维特征转换到 d_model 维
        self.face_feature_linear = nn.Linear(9, d_model)

        # 卷积层（共享权重）
        self.cube_net_conv = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.cube_net_activation = nn.ReLU()
        self.cube_net_dropout = nn.Dropout(dropout)

        self.src_emb_dropout = nn.Dropout(dropout)
        self.tgt_emb_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

        # 位置编码
        self.src_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.tgt_embedding = nn.Embedding(num_moves, d_model, padding_idx=19)
        self.tgt_pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, num_moves)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode_one_time_step(self, cube_state_t):
        """
        仅对单个时间步的cube_state做卷积特征提取
        输入: cube_state_t 形状 (B, 6, 3, 3)
        输出: (B, d_model) 对该时间步的聚合特征
        """
        B = cube_state_t.size(0)
        # 1) face-level特征 (每个面先用 face_feature_linear 做9->d_model的映射)
        #   cube_state_t.shape = (B, 6, 3, 3) -> reshape成(B, 6, 9), 然后线性
        face_flat = cube_state_t.view(B, 6, 9).float()  # (B, 6, 9)
        face_feat_linear = self.face_feature_linear(face_flat)  # (B, 6, d_model)

        # 2) 卷积特征
        #   再次 reshape -> (B*6, 1, 3, 3), 卷积通道是1
        face_reshaped = cube_state_t.view(B*6, 1, 3, 3)
        conv_out = self.cube_net_conv(face_reshaped)  # (B*6, d_model, 3, 3)
        conv_out = self.cube_net_activation(conv_out)
        conv_out = self.cube_net_dropout(conv_out)
        #   平均池化
        conv_face_feat = conv_out.view(B*6, self.d_model, -1).mean(dim=2)  # (B*6, d_model)
        conv_face_feat = conv_face_feat.view(B, 6, self.d_model)  # (B, 6, d_model)

        # 3) 融合 face_feat_linear & conv_face_feat
        fused_face_feat = 0.5 * (face_feat_linear + conv_face_feat)  # (B, 6, d_model)

        # 4) 最简单的聚合: 对6个面取平均
        time_step_feat = fused_face_feat.mean(dim=1)  # (B, d_model)

        return time_step_feat

    def forward(self, src, tgt_input):
        """
        src: shape (B, src_seq_len, input_dim=55)
        tgt_input: shape (B, tgt_seq_len-1)
        """
        B, src_seq_len, _ = src.shape
        # src 的前54维是6个面(6*9=54), 最后1维是 move 信息?

        # ========== 在时间维度上循环，每一个时间步都独立做卷积特征提取 ==========
        # 维度 (B, src_seq_len, 6, 3, 3)
        cube_states_6faces = src[..., :54].view(B, src_seq_len, 6, 3, 3)

        # 对每个时间步做 encode, 并将输出收集成列表
        time_step_feats = []
        for t in range(src_seq_len):
            # 取第 t 个时间步 shape (B, 6, 3, 3)
            cube_state_t = cube_states_6faces[:, t, :, :, :]
            # 做卷积 -> (B, d_model)
            feat_t = self.encode_one_time_step(cube_state_t)
            time_step_feats.append(feat_t)

        # 拼回 (src_seq_len, B, d_model)
        # 这样可以符合transformer输入 (T, B, d_model)
        src_encoded = torch.stack(time_step_feats, dim=0)  # => (src_seq_len, B, d_model)

        # 加上 src 的位置编码
        positions = torch.arange(src_seq_len, device=src.device).unsqueeze(1)  # => (T, 1)
        src_pos_embed = self.src_pos_embedding(positions)  # => (T, d_model)
        src_pos_embed = src_pos_embed.unsqueeze(1)  # => (T, 1, d_model)
        src_encoded = src_encoded + src_pos_embed  # broadcasting
        src_encoded = self.src_emb_dropout(src_encoded)  # => (T, B, d_model)

        # -------- Decoder Embedding --------
        tgt_input = tgt_input.permute(1, 0)  # => (tgt_seq_len-1, B)
        tgt_emb = self.tgt_embedding(tgt_input)  # => (tgt_seq_len-1, B, d_model)
        tgt_positions = torch.arange(tgt_emb.size(0), device=tgt_emb.device).unsqueeze(1)
        tgt_pos_embed = self.tgt_pos_embedding(tgt_positions)  # => (tgt_seq_len-1, d_model)
        tgt_pos_embed = tgt_pos_embed.unsqueeze(1)  # => (tgt_seq_len-1, 1, d_model)
        tgt_emb = tgt_emb + tgt_pos_embed
        tgt_emb = self.tgt_emb_dropout(tgt_emb)

        # -------- Causal Mask --------
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)

        # -------- Transformer --------
        out = self.transformer(
            src=src_encoded,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
        )
        out = out.permute(1, 0, 2)  # => (B, tgt_seq_len-1, d_model)
        out = self.dropout1(out)
        logits = self.fc_out(out)
        return logits


if __name__ == "__main__":
    B = 2
    src_seq_len = 8  # 状态序列长度
    tgt_seq_len = 8  # move 序列长度
    input_dim = 55
    num_moves = 18

    model = RubikSeq2SeqTransformer(input_dim=input_dim, num_moves=num_moves)
    src = torch.randn(B, src_seq_len, input_dim)  # 魔方状态输入
    tgt = torch.randint(0, num_moves, (B, tgt_seq_len))  # move 序列（索引）

    logits = model(src, tgt)
    print(logits.shape)
