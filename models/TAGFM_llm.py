import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_mean, scatter_add
from transformers import AutoModel, AutoTokenizer
from models.graph_encoder import setup_graph_module
import torch.nn.functional as F


class StructureAwareMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, struct_dim=32, memory_size=64, memory_dim=768):
        super().__init__()

        # 结构特征编码层
        self.struct_encoder = nn.Sequential(
            nn.Linear(1, struct_dim),
            nn.ReLU()
        )

        # 主网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + struct_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 记忆模块
        self.memory = nn.Embedding(memory_size, memory_dim)
        self.loss_fn = nn.MSELoss()

    def aggre_edges(self, edge_index, edge_attr, x, num_nodes):
        row, col = edge_index
        edge_aggr = scatter_add(edge_attr, col, dim=0, dim_size=num_nodes)
        return x + edge_aggr

    def forward(self, batch, feature, edge_attr=None, global_ft=False):
        # 计算结构特征
        if edge_attr is not None:
            feature = self.aggre_edges(batch.edge_index, edge_attr, feature, feature.size(0))

        struct_feat = self.struct_encoder(batch.ppr.unsqueeze(-1))
        # 计算节点特征
        node_feats = self.mlp(torch.cat([feature, struct_feat], dim=-1))
        graph_feats = scatter_mean(node_feats, batch.batch, dim=0)
        pooled_feats = self.pool(graph_feats)

        # 记忆增强
        M = self.memory.weight  # (memory_size, memory_dim)
        M = M.expand(node_feats.size(0), -1, -1)  # (batch_size, memory_size, memory_dim)

        node_feats_unsq = node_feats.unsqueeze(2)  # (batch_size, output_dim, 1)
        mem_key = F.softmax(torch.matmul(M, node_feats_unsq), dim=1)  # (batch_size, memory_size, 1)
        re_node_feats = mem_key * M  # (batch_size, memory_size, memory_dim)
        re_node_feats = torch.mean(re_node_feats, dim=1)  # (batch_size, memory_dim)

        loss = self.loss_fn(node_feats, re_node_feats)

        if global_ft:
            return pooled_feats, node_feats, re_node_feats
        else:
            return pooled_feats, re_node_feats, loss


class TAGFM(nn.Module):
    def __init__(self, args, text_encoder) -> None:
        super().__init__()
        self.lm_model = text_encoder
        self.mlp = StructureAwareMLP(
            input_dim=text_encoder.config.hidden_size,
            hidden_dim=args.hidden_size,
            output_dim=args.hidden_size
        )

        self.gnn_teacher = self.init_gnn_encoder(args, text_encoder)
        self.freeze_teacher_params()

        self.lam = args.lam

        self.feature_projector = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.PReLU()
        )

        self.task_loss = nn.CrossEntropyLoss()

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

    def forward(self, batch, epoch=0):
        with torch.no_grad():
            mask_token_emb = self.lm_model.mask_encoder(batch.x)
        mask_cls_token_emb = mask_token_emb[:, 0, :]
        graph_mlp_emb, mlp_emb, memory_loss = self.mlp(batch,mask_cls_token_emb,batch.edge_attr)

        with torch.no_grad():
            token_emb = self.lm_model(batch.x)
            cls_token_emb = token_emb[:, 0, :]
            gnn_emb = self.gnn_teacher(
                x=cls_token_emb,
                edge_index=batch.edge_index
            )
            graph_gnn_emb = scatter_mean(gnn_emb, batch.batch, dim=0)

        loss_gnn = F.cosine_similarity(graph_mlp_emb, graph_gnn_emb).mean()

        mlp_emb_expanded = mlp_emb.unsqueeze(1).expand(-1, batch.x["input_ids"].shape[1], -1)
        fused_emb = torch.cat([mask_token_emb, mlp_emb_expanded], dim=-1)
        hidden_emb = self.feature_projector(fused_emb)

        mask = (batch.x["masked_input_ids"] == self.lm_model.mask_token_id)
        masked_hidden = hidden_emb[mask]
        masked_labels = batch.x["input_ids"][mask]
        with torch.no_grad():
            prediction_scores = self.lm_model.cls_head(masked_hidden)
        mask_loss = self.task_loss(prediction_scores, masked_labels.long())

        total_loss = mask_loss + loss_gnn + memory_loss
        return total_loss, loss_gnn.item()

    def freeze_teacher_params(self):
        for p in self.gnn_teacher.parameters():
            p.requires_grad_(False)
        for p in self.lm_model.parameters():
            p.requires_grad_(False)


    @torch.no_grad()
    def emb(self, batch):
        token_emb = self.lm_model(batch.x)
        return scatter_mean(token_emb[:, 0, :], batch.batch, dim=0)


    @torch.no_grad()
    def inference(self, batch):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)
            output,_, _ = self.mlp(batch,token_emb)
            return output



