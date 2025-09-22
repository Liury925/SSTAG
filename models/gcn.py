import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from utils.functions import create_activation, create_norm


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 activation,
                 feat_drop,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 extra_input_fc=False,
                 extra_output_fc=True,
                 **kwargs):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GCNConv(
                in_dim, out_dim,
                feat_drop=feat_drop,
                residual=last_residual,
                activation=last_activation,
                norm=last_norm))
        else:
            self.gcn_layers.append(GCNConv(
                in_dim, num_hidden,
                feat_drop=feat_drop,
                residual=residual,
                activation=create_activation(activation),
                norm=norm))

            for l in range(1, num_layers - 1):
                self.gcn_layers.append(GCNConv(
                    num_hidden, num_hidden,
                    feat_drop=feat_drop,
                    residual=residual,
                    activation=create_activation(activation),
                    norm=norm))

            self.gcn_layers.append(GCNConv(
                num_hidden, out_dim,
                feat_drop=feat_drop,
                residual=last_residual,
                activation=last_activation,
                norm=last_norm))

        if extra_input_fc:
            self.input_fc = nn.Linear(in_dim, num_hidden)
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
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            hidden_list.append(x)

        if self.head is not None:
            x = self.head(x)

        return (x, torch.cat(hidden_list, dim=-1)) if return_hidden else x

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GCNConv(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 residual=False,
                 activation=None,
                 norm=None,
                 bias=True):
        super(GCNConv, self).__init__(aggr='add')
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc = nn.Linear(in_feats, out_feats, bias=False)

        if residual:
            self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_fc = None

        self.norm = create_norm(norm)(out_feats) if norm else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)

    def forward(self, x, edge_index):
        # 添加自环并归一化
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 特征变换
        x = self.feat_drop(x)
        h = self.fc(x)

        # 传播并归一化
        h = self.propagate(edge_index, x=h, norm=norm)

        # 残差连接
        if self.res_fc is not None:
            h += self.res_fc(x)

        # 归一化
        if self.norm is not None:
            h = self.norm(h)

        # 激活函数
        if self.activation is not None:
            h = self.activation(h)

        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j