import torch.nn as nn
from .gat import GAT
from .gcn import GCN
from .gin import GIN
from utils.functions import create_norm


def setup_graph_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers,
                      dropout, activation, residual, norm, nhead=None, nhead_out=None,
                      attn_drop=None, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            feat_drop=dropout,
            residual=residual,
            norm=create_norm(norm),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            feat_drop=dropout,
            residual=residual,
            norm=create_norm(norm),
            mlp_layers=2,
            aggregator_type='sum'
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod