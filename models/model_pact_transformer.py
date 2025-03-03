import torch
import torch.nn as nn
import einops

class RubikGPT(nn.Module):
    """
    一个示例模型：接收 shape = (B, T, 55) 的输入，其中:
      - 0:54 (共 54 维) 表示魔方状态
      - 54:55 (共 1 维) 表示 action (动作)
    并将它们分别投影到 d_model 维度后，拼到一起做 GPT-like 的自回归建模。
    """

    def __init__(self,
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 max_seq_len=100,     # 最长时间序列 (history_len+1) 的估计
                 # 你可以根据自己需求调整 feedforward、dropout、num_layers 等
                 ff_dim=512,         # feedforward 内部层维度
                 dropout=0.1,
                 vocab_size=21       # 假设我们要预测 21 种离散动作,仅作演示
                 ):
        super().__init__()
        self.d_model = d_model

        # 1) 分别定义对 state(54维) 和 action(1维) 的线性映射
        self.state_emb = nn.Linear(54, d_model)
        self.action_emb = nn.Linear(1,  d_model)

        # 2) 定义位置编码（PACT风格：global + local）
        #    - global_pos_embed: (1, 2*T, d_model) => 用于区分每一个 token 的全局位置
        #    - local_pos_embed:  (1, T, d_model)   => 用于区分每个 state-action 对在时间序列中的相对位置
        #      然后再 repeat_interleave(2, dim=1) 使其匹配 (1, 2*T, d_model)
        self.pos_embd_global = nn.Embedding(max_seq_len * 2, d_model)
        self.pos_embd_local  = nn.Embedding(max_seq_len,     d_model)

        # 3) 用 nn.TransformerDecoder 来模拟一个简单 GPT-like 的自回归结构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu'
        )
        self.gpt = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        # 4) 输出层：如果要预测一个离散动作的分布，则投影到 vocab_size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        """
        src: shape = (B, T, 55)
          - 其中 src[:,:, :54] => 魔方状态
          -      src[:,:, 54] => 动作
        返回:
          - logits: (B, 2*T, vocab_size)，表示对于每个 state, action 的 token，我们预测什么
                    （只是个示例，也可以只在 action token 上预测）
        """
        B, T, _ = src.shape

        # 1) 拆分 state/action
        #    state => (B,T,54), action => (B,T,1)
        state  = src[:, :, :54].float()
        action = src[:, :, 54:].float()   # 最后一维 (B,T,1)

        # 2) 分别映射到 d_model 维度
        state_emb  = self.state_emb(state)    # => (B,T,d_model)
        action_emb = self.action_emb(action)  # => (B,T,d_model)

        # 3) 拼接 token：先 stack => (B,T,2,d_model)，再 reshape => (B,2T,d_model)
        #    其中每个时间步 t，会依次是 state_emb[t], action_emb[t]
        combined_emb = torch.stack([state_emb, action_emb], dim=2)  # (B, T, 2, d_model)
        combined_emb = einops.rearrange(combined_emb, "b t two d -> b (t two) d")  # (B, 2T, d_model)

        # 4) 准备位置编码
        #    - global: pos_global = range(0, 2T)
        #    - local:  pos_local  = range(0, T) => repeat_interleave(2, dim=1)
        pos_global_ids = torch.arange(0, 2*T, device=src.device).unsqueeze(0)  # (1, 2T)
        pos_local_ids  = torch.arange(0, T,   device=src.device).unsqueeze(0)  # (1, T)
        global_emb = self.pos_embd_global(pos_global_ids)  # (1, 2T, d_model)
        local_emb  = self.pos_embd_local(pos_local_ids)    # (1, T, d_model)
        local_emb  = local_emb.repeat_interleave(2, dim=1) # => (1, 2T, d_model)

        combined_emb = combined_emb + global_emb + local_emb  # (B, 2T, d_model)

        # 5) 因果 Mask：防止看到后面token
        #    nn.TransformerDecoder 需要 (L, L) 的 mask, 其中 L=2T
        combined_emb = einops.rearrange(combined_emb, "b seq d -> seq b d")  # => (2T, B, d_model)
        seq_len = combined_emb.size(0)
        # causal_mask: 上三角为 -inf (阻止访问未来token)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=src.device),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))

        # 6) 调用 TransformerDecoder
        #    因为是 GPT-like，自回归，所以 memory 我们可以弄个假的全零即可
        fake_memory = torch.zeros(1, B, self.d_model, device=src.device)  # (1,B,d_model) 仅演示
        out = self.gpt(
            tgt=combined_emb,
            memory=fake_memory,
            tgt_mask=causal_mask
        )  # => (2T, B, d_model)

        out = einops.rearrange(out, "seq b d -> b seq d")  # => (B, 2T, d_model)

        # 7) 映射到 vocab_size
        logits = self.fc_out(out)  # => (B, 2T, vocab_size)

        return logits


# ------------------- 测试一下 -------------------
if __name__ == "__main__":
    B = 2
    # history_len+1 = 5 比如
    T = 5
    dummy_src = torch.randn(B, T, 55)  # (B, T, 55)

    model = RubikGPT(d_model=64, nhead=4, num_layers=2, max_seq_len=10, ff_dim=128, vocab_size=21)
    output = model(dummy_src)  # => (B, 2*T, vocab_size)
    print("output shape =", output.shape)
    # 预期: (2, 10, 21)  (B=2, 2*T=10, vocab_size=21)
