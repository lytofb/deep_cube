import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 使用 PyG 中的 GCNConv

class SimpleGCN(nn.Module):
    """
    使用 PyTorch Geometric 的 GCNConv 构造的简单 GCN 模型，
    对每个图（魔方状态，共 54 个节点）进行卷积，然后全局均值池化得到图表示。
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
        Args:
            x: 节点特征矩阵，形状 (N, in_dim) ，其中 N=54
            edge_index: 边索引，形状 (2, E)
        Returns:
            图表示，形状 (1, out_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        # 对所有节点取均值，得到 (1, out_dim) 的图表示
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
        """
        # 转换为 tensor（若不是），并确保最后一维为 54
        if not torch.is_tensor(s6x9):
            s6x9 = torch.as_tensor(s6x9, dtype=torch.long)
        if s6x9.shape[-1] != 54:
            raise ValueError("The last dimension of s6x9 must be 54.")

        # 保存原始批量结构
        original_shape = s6x9.shape[:-1]
        s6x9_flat = s6x9.reshape(-1, 54)  # 形状 (N, 54)，其中 N 是状态总数

        outputs = []
        for state in s6x9_flat:
            state = state.to(self.device)
            # 颜色 embedding 得到节点特征：形状 (54, embed_dim)
            node_colors = torch.as_tensor(state, dtype=torch.long, device=self.device)
            x = self.color_emb(node_colors)
            # 输入 GCN 得到图表示：形状 (1, out_dim)
            out = self.gnn(x, self.edge_index)
            outputs.append(out.squeeze(0))  # squeeze 成 (out_dim,)
        out_tensor = torch.stack(outputs, dim=0)  # (N, out_dim)
        # 恢复原始批量结构
        out_tensor = out_tensor.reshape(*original_shape, self.out_dim)
        return out_tensor

    def encode_move(self, mv):
        # 延迟导入，避免循环依赖
        from utils import move_str_to_idx,PAD_TOKEN
        if mv is None:
            return PAD_TOKEN
        return move_str_to_idx(mv)
