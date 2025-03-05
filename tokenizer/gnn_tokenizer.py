# gnn_tokenizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import move_str_to_idx

PAD_TOKEN = -1  # 仅作示例，用于 encode_move
# 你可以把这和你的 dataset/dict constants 整合

class SimpleGCNLayer(nn.Module):
    """简化版 GCN 层: X_new = D^{-1/2} A D^{-1/2} X W"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, edge_index):
        N = x.size(0)
        row, col = edge_index
        # 加self-loop
        loop_idx = torch.arange(N, device=x.device)
        loop_edges = torch.stack([loop_idx, loop_idx], dim=0)
        full_edge_index = torch.cat([edge_index, loop_edges], dim=1)

        # 计算度数
        row2, col2 = full_edge_index
        deg = torch.bincount(row2, minlength=N).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        # 线性变换
        x = self.linear(x)

        # 消息传递
        out = torch.zeros_like(x)
        norm = deg_inv_sqrt[row2] * deg_inv_sqrt[col2]
        out.index_add_(0, row2, norm.unsqueeze(-1)* x[col2])
        return out


class SimpleGCN(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, out_dim=128, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        # 第1层
        self.layers.append(SimpleGCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.layers.append(SimpleGCNLayer(hidden_dim, out_dim))
        else:
            self.layers[0] = SimpleGCNLayer(in_dim, out_dim)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers)-1:
                x = F.relu(x)
        # x.shape = (N, out_dim), N=节点数=54
        # 做 pooling => (1, out_dim)
        out = x.mean(dim=0, keepdim=True)
        return out  # shape (1, out_dim)


class GNNTokenizer:
    """
    将魔方状态(贴纸颜色)转换成一个 d_model 维度 embedding，
    并且对动作字符串也做 encode_move (保留或改写与原先兼容的逻辑)。
    """
    def __init__(self, edge_index, num_colors=6,
                 gnn_hidden=64, gnn_out=128, gnn_layers=2):
        """
        Args:
            edge_index: (2,E) 预先定义好的魔方贴纸邻接关系
            num_colors: 贴纸颜色总数(如 6)
            gnn_hidden, gnn_out: GCN hidden/out 维度
            gnn_layers: GCN层数
        """
        self.edge_index = edge_index  # (2,E)
        self.num_colors = num_colors

        # 颜色embedding: 把每个节点颜色ID [0..5] => embed_dim
        self.embed_dim = 8  # 贴纸颜色的嵌入尺寸(示例)
        self.color_emb = nn.Embedding(num_embeddings=num_colors,
                                      embedding_dim=self.embed_dim)

        self.gnn = SimpleGCN(
            in_dim=self.embed_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            num_layers=gnn_layers
        )

        self.out_dim = gnn_out  # 最终图表示维度

    def encode_state(self, s6x9):
        """
        s6x9: 6x9 矩阵, 或 54 list, 存储颜色ID [0..5] (示例)
        返回: shape (d_model,) 的向量 (或者 shape(1, d_model))
        """
        # 1) 将 6x9 => node_colors: (N,)
        #    如果你现在还用 convert_state_to_tensor，则先把贴纸映射到 [0..5].
        #    这里示例直接把 s6x9 flatten 后当 colors
        node_colors = torch.tensor(s6x9, dtype=torch.long)  # shape=(54,) (示例)

        # 2) color_emb => shape=(54, embed_dim)
        x = self.color_emb(node_colors)

        # 3) gnn => (1, out_dim)
        out = self.gnn(x, self.edge_index)
        return out.squeeze(0)  # => shape (out_dim,)

    def encode_move(self, mv):
        # 可保持原先逻辑: if None => PAD, else => move_str_to_idx
        if mv is None:
            return PAD_TOKEN
        # 这里示例: 'U','U2','U'... => int
        return move_str_to_idx(mv)  # 你已有 move_str_to_idx

