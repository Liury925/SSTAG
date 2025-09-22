from utils.functions import create_activation, create_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 shared_layers=False,
                 extra_input_fc=False,
                 extra_output_fc=True):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_dim, out_dim, 1, feat_drop, attn_drop, negative_slope, last_residual,
                norm=last_norm, concat_out=concat_out))
        else:
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual,
                create_activation(activation), norm=norm, concat_out=concat_out))

            for l in range(1, num_layers - 1):
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual,
                    create_activation(activation), norm=norm, concat_out=concat_out))

            self.gat_layers.append(GATConv(
                num_hidden * nhead, out_dim, 1, feat_drop, attn_drop, negative_slope, last_residual,
                activation=last_activation, norm=last_norm, concat_out=concat_out))

        if extra_input_fc:
            self.input_fc = nn.Linear(in_dim, num_hidden * nhead)
        else:
            self.input_fc = None

        if extra_output_fc:
            self.head = nn.Linear(out_dim, out_dim)
        else:
            self.head = None

    def aggre_edges(self, edge_index, edge_attr, x, num_nodes):
        row, col = edge_index
        edge_aggr = scatter_add(edge_attr, col, dim=0, dim_size=num_nodes)
        return x + edge_aggr

    def forward(self, x, edge_index, edge_attr=None, return_hidden=False):
        if edge_attr is not None:
            x = self.aggre_edges(edge_index, edge_attr, x, x.size(0))

        if self.input_fc is not None:
            x = self.input_fc(x)

        hidden_list = [x]
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            hidden_list.append(x)

        if self.head is not None:
            x = self.head(x)

        return (x, torch.cat(hidden_list, dim=-1)) if return_hidden else x


class GATConv(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__(aggr='add', node_dim=0)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.concat_out = concat_out
        self.negative_slope = negative_slope

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, num_heads, out_feats * 2))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act = activation
        self.norm = create_norm(norm)(out_feats * num_heads) if norm else None

        if residual:
            self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        else:
            self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.attn)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight)

    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 特征变换
        h = self.feat_drop(x)
        h = self.fc(h).view(-1, self.num_heads, self.out_feats)

        # 计算注意力系数
        row, col = edge_index
        alpha = (torch.cat([h[row], h[col]], dim=-1) * self.attn).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, col, num_nodes=x.size(0))
        alpha = self.attn_drop(alpha)

        # 消息传递
        out = self.propagate(edge_index, x=h, alpha=alpha)

        # 残差连接
        if self.res_fc is not None:
            res = self.res_fc(x).view(-1, self.num_heads, self.out_feats)
            out += res

        # 归一化
        if self.norm is not None:
            out = self.norm(out.view(-1, self.num_heads * self.out_feats))

        # 激活函数
        if self.act is not None:
            out = self.act(out)

        return out.view(-1, self.num_heads * self.out_feats)

    def message(self, x_j, alpha):
        return x_j * alpha.view(-1, self.num_heads, 1)