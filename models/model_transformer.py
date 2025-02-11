# models/model_transformer.py
import torch
import torch.nn as nn


class RubikTransformer(nn.Module):
    """
    把 54 贴纸视为 54 tokens，每个 token embedding dim = d_model。
    用 TransformerEncoder 处理，然后 pool/取cls 做分类。
    """

    def __init__(self, num_colors=6, d_model=64, nhead=4, num_layers=2, num_moves=18):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(num_colors, d_model)
        # 简单固定的位置编码 or learnable
        self.pos_embedding = nn.Embedding(54, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最后用一个线性输出
        self.fc_out = nn.Linear(d_model, num_moves)

    def forward(self, x):
        """
        x shape = (B, 54). Batch of 54-token sequences
        """
        B = x.size(0)

        # embed shape = (B,54,d_model)
        emb = self.embedding(x)
        # add pos encoding
        positions = torch.arange(0, 54, device=x.device).unsqueeze(0)  # (1,54)
        pos_emb = self.pos_embedding(positions)  # (1,54,d_model)
        emb = emb + pos_emb  # (B,54,d_model)

        # nn.TransformerEncoder 需要 shape=(seq_len, batch, d_model)
        emb = emb.permute(1, 0, 2)  # => (54,B,d_model)

        encoded = self.transformer_encoder(emb)  # (54,B,d_model)
        # 把 seq_dim 平均或取[CLS], 这里简单平均
        encoded_mean = encoded.mean(dim=0)  # (B,d_model)

        logits = self.fc_out(encoded_mean)  # (B,18)
        return logits
