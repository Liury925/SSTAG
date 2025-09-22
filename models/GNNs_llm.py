import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_mean, scatter_add
from transformers import AutoModel, AutoTokenizer
from models.graph_encoder import setup_graph_module
import torch.nn.functional as F



class GNNs(nn.Module):
    def __init__(self, args, text_encoder,num_class) -> None:
        super().__init__()
        self.lm_model = text_encoder
        self.gnn = self.init_gnn_encoder(args, text_encoder)
        self.classifier = nn.Linear(args.hidden_size, num_class)
        self.task_loss = nn.CrossEntropyLoss()

    def init_gnn_encoder(self, args, text_encoder):
        return setup_graph_module(
            m_type=args.model_name,
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

    def forward(self, batch, epoch=0):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)
        gnn_emb = self.gnn(
                x=token_emb,
                edge_index=batch.edge_index
        )
        graph_gnn_emb = scatter_mean(gnn_emb, batch.batch, dim=0)
        output = self.classifier(graph_gnn_emb)

        loss = self.task_loss(output, batch.y)
        return loss, None

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
            output = self.classifier(graph_gnn_emb)
            return output



