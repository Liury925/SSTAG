import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_mean, scatter_add
from transformers import AutoModel, AutoTokenizer
from models.graph_encoder import setup_graph_module
import torch.nn.functional as F



class GraphCL(nn.Module):
    def __init__(self, args, text_encoder) -> None:
        super().__init__()
        self.lm_model = text_encoder
        self.gnn = self.init_gnn_encoder(args, text_encoder)

    def init_gnn_encoder(self, args, text_encoder):
        return setup_graph_module(
            m_type=args.gnn_type,
            enc_dec="encoding",
            in_dim=text_encoder.indim,
            num_hidden=args.hidden_size // args.nhead,
            out_dim=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            residual=True,
            norm=args.norm,
            nhead=args.nhead,
            attn_drop=args.dropout,
            negative_slope=args.negative_slope
        )

    def feature_augmentation(self, x, mask_prob=0.2):
        mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_prob)).float()
        return x * mask

    def forward(self, batch, edge_index1, edge_index2, epoch=0):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)

        # aug1 = self.feature_augmentation(token_emb)
        # aug2 = self.feature_augmentation(token_emb)

        gnn_emb1 = self.gnn(
                x=token_emb,
                edge_index=edge_index1
        )
        graph_gnn_emb1 = scatter_mean(gnn_emb1, batch.batch, dim=0)

        gnn_emb2 = self.gnn(
            x=token_emb,
            edge_index=edge_index2
        )
        graph_gnn_emb2 = scatter_mean(gnn_emb2, batch.batch, dim=0)

        loss1 = self.calc_contrastive_loss(graph_gnn_emb1, graph_gnn_emb2)
        loss2 = self.calc_contrastive_loss(graph_gnn_emb2, graph_gnn_emb1)

        return (loss1+loss2).mean(), None

    def get_sim(self,z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.T)
        return sim

    def calc_contrastive_loss(self, z1, z2):
        f = lambda x: torch.exp(x)
        refl_sim = f(self.get_sim(z1, z1))
        between_sim = f(self.get_sim(z1, z2))
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -torch.log(between_sim.diag() / x1)
        return loss



    @torch.no_grad()
    def emb(self, batch):
        token_emb = self.lm_model(batch.x)
        return scatter_mean(token_emb[:, 0, :], batch.batch, dim=0)


    @torch.no_grad()
    def inference(self, batch):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)
            emb = self.gnn(x=token_emb,
                        edge_index=batch.edge_index)
            graph_gnn_emb = scatter_mean(emb, batch.batch, dim=0)
            return graph_gnn_emb



