import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, AttentionalAggregation
from einops import rearrange
import math

class DeformableGraphGNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_classes, topk, stride, agg_type='bi-interaction'):
        super().__init__()
        self.topk = topk
        self.stride = stride
        self.scale = dim_hidden ** -0.5
        self.agg_type = agg_type

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.offset_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, topk * 2),  # 输出 dx, dy 偏移（连续值）
            nn.Sigmoid()
        )
        
        
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError

        # 注意维度
        self.linear = nn.Linear(dim_hidden * 2 if agg_type == 'sage' else dim_hidden, dim_hidden)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        self.att_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden // 2, 1)
        )
        self.readout = AttentionalAggregation(self.att_net)
        self.expand_alpha = nn.Parameter(torch.randn(1))

    def forward(self, x, coords):
        B, N, D = x.shape

        # # 1. 特征嵌入
        x = self._fc1(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(x)  # [B, N, D]
        e_t = self.W_tail(x)
        # e_h = x
        # e_t = x

        # 2. 预测 offset（单位为像素）
        offset = self.offset_net(e_h).reshape(B, N, self.topk, 2)  # [B, N, K, 2] 这里改了原本不是e_h
        #offset = offset * self.stride * 500
        expand_1 = 0.5 + torch.sigmoid(self.expand_alpha)
        expand_factor = torch.clamp(expand_1, max=1.0)
        offset = offset * self.stride * math.sqrt(N) * expand_factor
        #offset = offset * self.stride * math.sqrt(N)
        #print("Stride: " + str(self.stride))
        
        # 3. 偏移 + 找邻居（动态构图）
        query_coords = coords.unsqueeze(2) + offset  # [B, N, K, 2]
        dist = torch.cdist(query_coords.view(B, N * self.topk, 2), coords, p=2)  # [B, N*K, N]
        knn_index = dist.argmin(dim=-1).view(B, N, self.topk)  # [B, N, K]

        # 4. 获取邻居特征
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(knn_index.device)  # [B, 1, 1]
        Nb_h = e_t[batch_indices, knn_index, :]  # [B, N, K, D]

        #5. 构造 edge 权重（MLP输入：e_h + Nb_h + offset）
        # h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)  # [B, N, K, D]
        # feat_pair = torch.cat([h_expand, Nb_h, offset], dim=-1)  # [B, N, K, 2D+2]，这个和头结点，尾节点和边属性有关
        # raw_score = self.edge_weight_net(feat_pair).squeeze(-1)  # [B, N, K]
        # edge_weight = F.softmax(raw_score, dim=-1)  # [B, N, K] 对每个节点的K个邻居归一化，得到重要性
        e_h_norm = F.normalize(e_h, dim=-1)
        Nb_h_norm = F.normalize(Nb_h, dim=-1)
        h_expand = e_h_norm.unsqueeze(2).expand(-1, -1, self.topk, -1)
        sim_score = torch.sum(h_expand * Nb_h_norm, dim=-1)
        edge_weight = F.softmax(sim_score, dim=-1)

        # 6. Gated knowledge attention
        eh_r = edge_weight.unsqueeze(-1) * Nb_h  # [B, N, K, D]
        gate = torch.tanh(h_expand + eh_r)  # [B, N, K, D]
        ka_weight = torch.einsum("bnkd,bnkd->bnk", Nb_h, gate)  # [B, N, K]
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(2)  # [B, N, 1, K]
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(2)  # [B, N, D]

        # 7. 聚合邻居特征（GCN/SAGE）
        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        else:
            raise NotImplementedError

        embedding = self.activation(self.linear(embedding))  # [B, N, D]
        h = self.message_dropout(embedding)

        # 8. 全局池化 + 分类
        h = self.readout(h.squeeze(0))  # [D]
        h = self.norm(h)
        return self.fc(h)  # [n_classes]

