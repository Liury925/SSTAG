import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MessagePassing
from utils.functions import create_activation, create_norm


class GIN(nn.Module):
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
                 eps=0.,
                 mlp_layers=2,
                 aggregator_type='sum',
                 **kwargs):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out
        self.eps = eps
        self.mlp_layers = mlp_layers

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        # Layer construction
        if num_layers == 1:
            self.gin_layers.append(GINConv(
                apply_func=make_mlp(in_dim, out_dim, num_hidden,
                                    mlp_layers, activation, feat_drop, last_norm),
                aggregator_type=aggregator_type,
                eps=eps,
                residual=last_residual))
        else:
            # Input layer
            self.gin_layers.append(GINConv(
                apply_func=make_mlp(in_dim, num_hidden, num_hidden,
                                    mlp_layers, activation, feat_drop, norm),
                aggregator_type=aggregator_type,
                eps=eps,
                residual=residual))

            # Hidden layers
            for _ in range(1, num_layers - 1):
                self.gin_layers.append(GINConv(
                    apply_func=make_mlp(num_hidden, num_hidden, num_hidden,
                                        mlp_layers, activation, feat_drop, norm),
                    aggregator_type=aggregator_type,
                    eps=eps,
                    residual=residual))

            # Output layer
            self.gin_layers.append(GINConv(
                apply_func=make_mlp(num_hidden, out_dim, num_hidden,
                                    mlp_layers, activation, feat_drop, last_norm),
                aggregator_type=aggregator_type,
                eps=eps,
                residual=last_residual))

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
        for layer in self.gin_layers:
            x = layer(x, edge_index)
            hidden_list.append(x)

        if self.head is not None:
            x = self.head(x)

        return (x, torch.cat(hidden_list, dim=-1)) if return_hidden else x

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(MessagePassing):
    def __init__(self,
                 apply_func,
                 aggregator_type='sum',
                 eps=0.,
                 residual=False):
        super().__init__(aggr='add')
        self.apply_func = apply_func
        self.aggregator_type = aggregator_type
        self.eps = eps

        if residual:
            in_dim = apply_func.input_dim
            out_dim = apply_func.output_dim
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

    def forward(self, x, edge_index):
        # 聚合邻居信息
        if self.aggregator_type == 'sum':
            neigh = self.propagate(edge_index, x=x)
        elif self.aggregator_type == 'mean':
            neigh = self.propagate(edge_index, x=x, aggr='mean')
        else:
            raise ValueError("Invalid aggregator type")

        # 中心节点缩放
        out = (1 + self.eps) * x + neigh

        # 应用MLP
        if self.apply_func is not None:
            out = self.apply_func(out)

        # 残差连接
        if self.res_fc is not None:
            out += self.res_fc(x)

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        if aggr is None:
            aggr = self.aggr
        return scatter_mean(inputs, index, dim=self.node_dim, dim_size=dim_size) if aggr == 'mean' else \
            scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)


def make_mlp(input_dim, output_dim, hidden_dim,
             num_layers, activation, dropout, norm=None):
    layers = []
    current_dim = input_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if norm is not None:
            layers.append(create_norm(norm)(hidden_dim))
        layers.append(create_activation(activation))
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))

    # 添加维度属性
    mlp = nn.Sequential(*layers)
    mlp.input_dim = input_dim
    mlp.output_dim = output_dim
    return mlp