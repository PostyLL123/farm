import torch
import torch.nn as nn

class GCN(nn.Module):
    """
    简单的图卷积层 (Graph Convolutional Layer)
    公式: H_out = A * H_in * W + b
    """
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        # 权重矩阵 W [in_features, out_features]
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 偏置 b [out_features]
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: 输入特征 [Batch, Nodes, In_Features]
            adj: 邻接矩阵 [Nodes, Nodes]
        Returns:
            output: [Batch, Nodes, Out_Features]
        """
        Batch, Nodes, Features = x.shape
        
        # 1. 空间聚合: AX (Graph Convolution part 1)
        # adj 是 (N, N), x 是 (B, N, F)
        # 我们需要将 adj 扩展为 (B, N, N) 或者利用广播机制
        # torch.matmul((N, N), (B, N, F)) 在 PyTorch 中可能不直接支持广播，
        # 最稳健的方法是使用 einsum 或先 unsqueeze
        
        # 使用 einsum 进行矩阵乘法: 'nm, bmf -> bnf'
        # n=Nodes, m=Nodes, b=Batch, f=In_Features
        # 这相当于对每个 batch 进行 adj * x
        support = torch.einsum('nm, bmf -> bnf', adj, x)
        
        # 2. 特征变换: (AX)W (Graph Convolution part 2)
        # support: [B, N, In_F], weights: [In_F, Out_F]
        output = torch.matmul(support, self.weights)
        
        # 3. 加偏置
        output = output + self.bias
        
        return output

class TGCNCell(nn.Module):
    """
    T-GCN 单元: 结合了 GCN 和 GRU
    """
    def __init__(self, input_dim, hidden_dim):
        super(TGCNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU 的门控机制需要输入 (input + hidden) 的维度
        # 我们将 input 和 hidden 拼接，然后通过 GCN
        
        # 更新门 (Update Gate) 的 GCN
        self.gcn_z = GCN(input_dim + hidden_dim, hidden_dim)
        # 重置门 (Reset Gate) 的 GCN
        self.gcn_r = GCN(input_dim + hidden_dim, hidden_dim)
        # 候选隐藏状态 (Candidate Hidden) 的 GCN
        self.gcn_h = GCN(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h, adj):
        """
        Args:
            x: 当前时刻输入 [Batch, Nodes, Input_Dim]
            h: 上一时刻隐藏状态 [Batch, Nodes, Hidden_Dim]
            adj: 邻接矩阵 [Nodes, Nodes]
        """
        # 1. 拼接输入和隐藏状态
        # shape: [Batch, Nodes, Input_Dim + Hidden_Dim]
        combined = torch.cat([x, h], dim=2)
        
        # 2. 计算更新门 Z (Sigmoid)
        z = torch.sigmoid(self.gcn_z(combined, adj))
        
        # 3. 计算重置门 R (Sigmoid)
        r = torch.sigmoid(self.gcn_r(combined, adj))
        
        # 4. 计算候选隐藏状态 h_hat (Tanh)
        # 重置门作用于旧的隐藏状态
        combined_r = torch.cat([x, r * h], dim=2)
        h_hat = torch.tanh(self.gcn_h(combined_r, adj))
        
        # 5. 计算最终隐藏状态
        h_new = z * h + (1 - z) * h_hat
        
        return h_new

class Model(nn.Module):
    """
    完整的 T-GCN 模型
    结构: T-GCN Cell (RNN loop) -> Output Layer (Linear)
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        
        # 从配置中读取超参数
        # 假设 cfg 是 argparse 的 Namespace 或 dict
        # 如果是 dict，请改用 cfg['key']
        self.input_dim = getattr(cfg, 'input_dim')  # 默认 11 (您的特征数)
        self.hidden_dim = getattr(cfg, 'hidden_dim') # 默认 64
        self.output_len = getattr(cfg, 'output_len') # 默认 12
        self.input_len = getattr(cfg, 'input_len')   # 默认 86
        
        # 1. T-GCN 单元
        self.tgcn_cell = TGCNCell(self.input_dim, self.hidden_dim)
        
        # 2. 输出层 (全连接层)
        # 将隐藏层维度映射到输出序列长度
        # 我们希望每个节点独立预测，所以是一个 shared Linear layer
        self.linear = nn.Linear(self.hidden_dim, self.output_len)

    def forward(self, x, adj):
        """
        Args:
            x: 历史输入序列 [Batch, Input_Len, Nodes, Input_Dim]
            adj: 邻接矩阵 [Nodes, Nodes] (或者是 [Batch, Nodes, Nodes])
        Returns:
            prediction: 未来预测序列 [Batch, Output_Len, Nodes]
        """
        Batch, Input_Len, Nodes, Feats = x.shape
        
        # 初始化隐藏状态 h0 为全零
        # shape: [Batch, Nodes, Hidden_Dim]
        h = torch.zeros(Batch, Nodes, self.hidden_dim).to(x.device)
        
        # 循环处理历史序列 (Encoding)
        for t in range(Input_Len):
            # 取出当前时刻 t 的数据: [Batch, Nodes, Input_Dim]
            x_t = x[:, t, :, :]
            h = self.tgcn_cell(x_t, h, adj)
        
        # 此时 h 包含了整个历史序列的时空信息
        
        # 通过输出层进行预测
        # h: [Batch, Nodes, Hidden_Dim] -> [Batch, Nodes, Output_Len]
        y_pred = self.linear(h)
        
        # 调整输出形状以匹配目标 [Batch, Output_Len, Nodes]
        # permute: (0, 2, 1) -> (Batch, Output_Len, Nodes)
        y_pred = y_pred.permute(0, 2, 1).unsqueeze(-1)
        
        # 如果需要 [Batch, Output_Len, Nodes, 1]，可以 unsqueeze
        # y_pred = y_pred.unsqueeze(-1)
        
        return y_pred