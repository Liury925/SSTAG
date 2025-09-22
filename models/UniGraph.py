import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer
from models.graph_encoder import setup_graph_module
from utils.functions import sce_loss
from torch_scatter import scatter_mean


class TAGFM(nn.Module):
    def __init__(self, args, text_encoder) -> None:
        super().__init__()
        self.device = args.device
        self.lam = args.lam
        self.momentum = args.momentum
        self.delayed_ema_epoch = args.delayed_ema_epoch
        self.mask_token_id = text_encoder.mask_token_id

        # 文本编码器
        self.lm_model = text_encoder

        # 图编码器初始化
        self.gnn_encoder = self.init_gnn_encoder(args, text_encoder)

        # 特征融合模块
        self.feature_projector = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.PReLU()
        )

        # 教师模型初始化
        if self.lam > 0:
            self.gnn_encoder_ema = self.init_gnn_encoder(args, text_encoder)
            self.gnn_encoder_ema.load_state_dict(self.gnn_encoder.state_dict())
            self.freeze_ema_params()

            self.predictor = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
            )

        # 损失函数
        self.criterion = CrossEntropyLoss()

    def init_gnn_encoder(self, args, text_encoder):
        """初始化PyG兼容的图编码器"""
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
        # 文本特征提取
        masked_token_emb = self.lm_model.mask_encoder(batch.x)
        masked_cls_token_emb = masked_token_emb.permute(1, 0, 2)[0]  # [CLS] token

        # 学生模型图编码
        gnn_emb = self.gnn_encoder(
            x=masked_cls_token_emb,
            edge_index=batch.edge_index
        )
        graph_gnn_emb = scatter_mean(gnn_emb, batch.batch, dim=0)

        # 教师模型处理
        loss_latent = torch.tensor(0.0, device=self.device)
        if self.lam > 0:
            with torch.no_grad():
                token_emb = self.lm_model(batch.x)
                cls_token_emb = token_emb.permute(1, 0, 2)[0]
                gnn_emb_ema = self.gnn_encoder_ema(
                    x=cls_token_emb,
                    edge_index=batch.edge_index
                )
                graph_gnn_emb_ema = scatter_mean(gnn_emb_ema, batch.batch, dim=0)

            pred = self.predictor(graph_gnn_emb)
            loss_latent = sce_loss(pred, graph_gnn_emb_ema.detach(), alpha=1)

        # 特征融合
        gnn_emb_expanded = gnn_emb.unsqueeze(1).expand(-1, batch.x["input_ids"].shape[1], -1)
        fused_emb = torch.cat([masked_token_emb, gnn_emb_expanded], dim=-1)
        hidden_emb = self.feature_projector(fused_emb)

        # MLM损失计算
        labels = batch.x["input_ids"].clone()
        labels[batch.x["masked_input_ids"] != self.mask_token_id] = -100
        prediction_scores = self.lm_model.cls_head(hidden_emb)
        mask_loss = self.criterion(
            prediction_scores.view(-1, self.lm_model.config.vocab_size),
            labels.view(-1))

        # 总损失
        total_loss = mask_loss + self.lam * loss_latent

        # EMA更新
        if self.lam > 0 and epoch >= self.delayed_ema_epoch:
            self.update_teacher_model()

        return total_loss, loss_latent

    def freeze_ema_params(self):
        """冻结教师模型参数"""
        for p in self.gnn_encoder_ema.parameters():
            p.requires_grad_(False)

    def update_teacher_model(self):
        """动量更新教师模型"""
        with torch.no_grad():
            m = self.momentum
            for param_q, param_k in zip(self.gnn_encoder.parameters(),
                                        self.gnn_encoder_ema.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def emb(self, batch):
        token_emb = self.lm_model(batch.x)
        return scatter_mean(token_emb[:, 0, :], batch.batch, dim=0) # 直接取CLS位置

    @torch.no_grad()
    def inference(self, batch):
        """图编码推理"""
        with torch.no_grad():
            token_emb = self.lm_model(batch.x, pooling=True)
            node_embs = self.gnn_encoder(token_emb,batch.edge_index)
            output = scatter_mean(node_embs, batch.batch, dim=0)
            return output
