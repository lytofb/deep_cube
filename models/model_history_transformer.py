import torch
import torch.nn as nn

class RubikHistoryTransformer(nn.Module):
    """
    用于 '历史序列 + 当前state' 的 Transformer 模型。
    假设输入 x.shape = (B, seq_len, 55)，
      其中 seq_len = history_len + 1，
      55 = 54(贴纸颜色) + 1(move 索引)。
      对于当前时刻只包含 state，可以把 move=-1 当占位。

    模型流程：
      1) 线性映射 input_dim -> d_model
      2) 加可学习位置编码
      3) TransformerEncoder
      4) 取最后一个 time step 的输出 -> 全连接 -> logits
    """
    def __init__(self,
                 input_dim=55,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 num_moves=18,
                 max_seq_len=20):
        """
        Args:
            input_dim: 每个时间步的特征维度 (默认55: 54贴纸 +1 move)
            d_model: Transformer的词向量维度
            nhead: 多头注意力的头数
            num_layers: TransformerEncoderLayer层数
            num_moves: 最终要分类的动作数量
            max_seq_len: 历史序列的最大长度(含当前), 用于可学习位置编码
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_moves = num_moves
        self.max_seq_len = max_seq_len

        # 1) 把 55 -> d_model 的线性映射
        self.linear_in = nn.Linear(input_dim, d_model)

        # 2) 可学习位置编码
        # 形状: (max_seq_len, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) 最后的分类层
        self.fc_out = nn.Linear(d_model, num_moves)

    def forward(self, x):
        """
        x.shape = (B, seq_len, input_dim)
          - B: batch_size
          - seq_len: history_len + 1
          - input_dim: 55

        Returns:
          logits.shape = (B, num_moves)
        """
        # x => (seq_len, B, input_dim) 以适配 PyTorch Transformer
        x = x.permute(1, 0, 2)  # (seq_len, B, input_dim)

        x = x.float()

        seq_len, batch_size, _ = x.shape

        # 1) 线性映射到 d_model => (seq_len, B, d_model)
        x = self.linear_in(x)

        # 2) 加可学习位置编码
        #    假设时序上第 i 个 time step 的位置是 i
        #    positions => shape (seq_len,), range(0..seq_len-1)
        positions = torch.arange(seq_len, device=x.device).long()  # (seq_len,)
        # pos_emb => (seq_len, d_model)
        # 需要确保 seq_len <= max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len({seq_len}) > max_seq_len({self.max_seq_len}). Need to increase max_seq_len.")
        pos_emb = self.pos_embedding(positions)  # (seq_len, d_model)

        # x + pos_emb(扩展batch维度): pos_emb => (seq_len,1,d_model)
        x = x + pos_emb.unsqueeze(1)  # 广播到 (seq_len,B,d_model)

        # 3) 送入 TransformerEncoder => (seq_len, B, d_model)
        out = self.transformer(x)  # (seq_len, B, d_model)

        # 4) 取最后一个 time step 的输出 => (B, d_model)
        last_out = out[-1, :, :]  # (B,d_model)

        logits = self.fc_out(last_out)  # (B, num_moves)
        return logits
