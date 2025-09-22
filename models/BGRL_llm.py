import copy
import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential
import torch
from torch.nn.functional import cosine_similarity
from torch_scatter import scatter_mean, scatter_add

class GraphSAGE_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
            ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def forward(self, x, edge_index, batch):
        h1 = self.convs[0](x, edge_index)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()


class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, args, text_encoder):
        super().__init__()
        # online network
        self.online_encoder = GraphSAGE_GCN(text_encoder.indim, args.hidden_size, args.hidden_size)
        self.predictor = MLP_Predictor(args.hidden_size, args.hidden_size)

        # target network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.lm_model = text_encoder

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def drop_features(self, features, drop_rate=0.3):
        drop_mask = torch.empty(
            (features.size(1),),
            dtype=torch.float32,
            device=features.device
        ).uniform_(0, 1) < drop_rate

        modified_features = features.clone()  # 关键修改点
        modified_features[:, drop_mask] = 0
        return modified_features

    def get_embedding(self,batch, edge_index1, edge_index2):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x, pooling=True)
        online_y = self.online_encoder(token_emb,edge_index1,batch.batch)
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            drop_token_embs = self.drop_features(token_emb)
            target_y = self.target_encoder(drop_token_embs,edge_index2,batch.batch).detach()

        return online_q, target_y


    def forward(self, batch, edge_index1, edge_index2, epoch=0):
        # forward online network
        q1, y2 = self.get_embedding(batch, edge_index1, edge_index2)
        q2, y1 = self.get_embedding(batch, edge_index2, edge_index1)
        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        return loss,None

    def inference(self, batch):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x, pooling=True)
            rep = self.online_encoder(token_emb,batch.edge_index,batch.batch)
            rep = scatter_mean(rep, batch.batch, dim=0)
            return rep


