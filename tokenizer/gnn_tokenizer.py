import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    """
    使用 PyTorch Geometric 的 GCNConv 构造的简单 GCN 模型，
    对单个图（魔方状态，共 54 个节点）进行卷积，原始实现内部做了全局均值池化。
    """

    def __init__(self, in_dim=8, hidden_dim=64, out_dim=128, num_layers=2):
        """
        Args:
            in_dim: 每个节点的输入特征维度（这里为颜色嵌入维度）
            hidden_dim: 隐藏层维度
            out_dim: 输出图表示维度
            num_layers: GCN 层数
        """
        super(SimpleGCN, self).__init__()
        self.convs = nn.ModuleList()
        # 第一层：从 in_dim 到 hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # 最后一层：输出到 out_dim
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            self.convs[0] = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        仅适用于单图输入，内部对所有节点取均值，返回 (1, out_dim) 的图表示。
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        out = x.mean(dim=0, keepdim=True)
        return out


class GNNTokenizer:
    """
    使用 PyTorch Geometric 重构的 GNNTokenizer：
      - 将魔方状态（贴纸颜色）通过颜色 embedding 后输入 GCN 得到图表示，
      - 同时保留对动作字符串的 encode_move 功能。
    """

    def __init__(self, edge_index, num_colors=6,
                 gnn_hidden=64, gnn_out=128, gnn_layers=2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # 将预先定义好的边索引放到指定设备上
        self.edge_index = edge_index.to(self.device)
        self.num_colors = num_colors

        # 定义颜色 embedding，将颜色 ID 映射到 embed_dim（示例取 8）
        self.embed_dim = 8
        self.color_emb = nn.Embedding(num_embeddings=num_colors,
                                      embedding_dim=self.embed_dim).to(self.device)

        # 使用 PyG 实现的 GCN 进行图表示学习
        self.gnn = SimpleGCN(in_dim=self.embed_dim,
                             hidden_dim=gnn_hidden,
                             out_dim=gnn_out,
                             num_layers=gnn_layers).to(self.device)

        self.out_dim = gnn_out

    def encode_state(self, s6x9):
        """
        输入：
          s6x9: 魔方状态，可以是一个 6x9 的矩阵或 54 长的列表，存储颜色ID（范围 0~5），也支持批量输入，形状为 (..., 54)
        输出：
          每个状态经过颜色 embedding 和 GCN 后得到图表示，形状为 (..., out_dim)

        这里我们采用批处理方式：
          1. 将输入 reshape 成 (N, 54)，N 为状态总数；
          2. 对所有状态同时做颜色 embedding，得到 (N*54, embed_dim)；
          3. 构造 batched 边索引，每个状态对应的边索引加上偏移量（i*54）；
          4. 使用 GCN 层计算所有节点特征，然后对每个状态（54 个节点）取均值作为图表示；
          5. 恢复原始的 batch 维度。
        """
        # 转换为 tensor（若不是），并确保最后一维为 54
        if not torch.is_tensor(s6x9):
            s6x9 = torch.as_tensor(s6x9, dtype=torch.long)
        if s6x9.shape[-1] != 54:
            raise ValueError("The last dimension of s6x9 must be 54.")

        # 保存原始批量结构，并展开为 (N, 54)
        original_shape = s6x9.shape[:-1]
        s6x9_flat = s6x9.reshape(-1, 54)  # (N, 54)
        num_states = s6x9_flat.shape[0]

        # 构造 batched 边索引
        # 原始 edge_index 对应单个图，形状 (2, E)
        base_edge_index = self.edge_index  # (2, E)
        E = base_edge_index.shape[1]
        # 对每个状态加上偏移量 i*54
        batch_edge_indices = []
        for i in range(num_states):
            offset = i * 54
            batch_edge_indices.append(base_edge_index + offset)
        # 拼接成 batched edge_index，形状 (2, num_states*E)
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)

        # 构造 batch 向量，记录每个节点所属图编号
        batch_vector = torch.arange(num_states, device=self.device).unsqueeze(1).repeat(1, 54).reshape(-1)

        # 颜色 embedding：将 (N, 54) 转换为 (N, 54, embed_dim)，再 reshape 为 (N*54, embed_dim)
        s6x9_flat = s6x9_flat.to(self.device, dtype=torch.long)
        x = self.color_emb(s6x9_flat)  # (N, 54, embed_dim)
        x = x.reshape(-1, self.embed_dim)  # (N*54, embed_dim)

        # 使用 GCN 层对 batched 图进行前向传播（注意，此处不调用 self.gnn.forward，因为其内部做了整体 pooling）
        # 我们直接对每个卷积层逐层计算
        for i, conv in enumerate(self.gnn.convs):
            x = conv(x, batch_edge_index)
            if i < len(self.gnn.convs) - 1:
                x = F.relu(x)
        # 此时 x 的形状为 (N*54, out_dim)

        # 对每个状态（54 个节点）做均值池化，恢复成 (N, out_dim)
        x = x.reshape(num_states, 54, -1).mean(dim=1)  # (N, out_dim)

        # 恢复原始的 batch 结构
        out_tensor = x.reshape(*original_shape, self.out_dim)
        return out_tensor

    def encode_move(self, mv):
        # 延迟导入，避免循环依赖
        from utils import move_str_to_idx, PAD_TOKEN
        if mv is None:
            return PAD_TOKEN
        return move_str_to_idx(mv)
