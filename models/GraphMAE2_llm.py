from itertools import chain
from typing import Optional
import torch
import torch.nn as nn
from functools import partial
from utils.functions import sce_loss
from models.graph_encoder import setup_graph_module
from torch_scatter import scatter_mean, scatter_add

class GraphMAE2(nn.Module):
    def __init__(
            self,
            args,text_encoder,
            num_remasking: int=3,
            mask_rate: float = 0.3,
            remask_rate: float = 0.5,
            remask_method: str = "random",
            mask_method: str = "random",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            alpha_l: float = 2,
            lam: float = 1.0,
            delayed_ema_epoch: int = 0,
            momentum: float = 0.996,
            replace_rate: float = 0.0,
            zero_init: bool = False,
    ):
        super(GraphMAE2, self).__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l
        self._delayed_ema_epoch = delayed_ema_epoch

        self.num_remasking = num_remasking
        self._drop_edge_rate = drop_edge_rate
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam

        self.lm_model = text_encoder
        self.gnn = self.init_gnn_encoder(args, text_encoder)

        # build encoder
        self.encoder = self.init_gnn_encoder(args, text_encoder)
        self.decoder = self.init_gnn_encoder(args, text_encoder)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, text_encoder.indim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, args.hidden_size))

        self.encoder_to_decoder = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        if not zero_init:
            self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.projector = nn.Sequential(
            nn.Linear(args.hidden_size, 256),
            nn.PReLU(),
            nn.Linear(256, args.hidden_size),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(args.hidden_size, 256),
            nn.PReLU(),
            nn.Linear(256, args.hidden_size),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )

        self.encoder_ema = self.init_gnn_encoder(args, text_encoder)
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()

        self.print_num_parameters()

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

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(
            f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)


    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion


    def forward(self, batch, epoch=0):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)

        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(token_emb, self._mask_rate)
        enc_rep = self.encoder(x=use_x, edge_index=batch.edge_index)

        with torch.no_grad():
            latent_target = self.encoder_ema(x=token_emb, edge_index=batch.edge_index)
            latent_target = self.projector_ema(latent_target[keep_nodes])

        latent_pred = self.projector(enc_rep[keep_nodes])
        latent_pred = self.predictor(latent_pred)
        loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(rep, self._remask_rate)
                recon = self.decoder(x=rep, edge_index=batch.edge_index)

                x_init = token_emb[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(origin_rep, mask_nodes)
            x_rec = self.decoder(x=rep, edge_index=batch.edge_index)[mask_nodes]
            x_init = token_emb[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return loss, None

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
                # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def inference(self, batch):
        with torch.no_grad():
            token_emb = self.lm_model(batch.x,pooling=True)
            rep = self.encoder(x=token_emb,
                        edge_index=batch.edge_index)
            graph_gnn_emb = scatter_mean(rep, batch.batch, dim=0)
            return graph_gnn_emb

    def get_encoder(self):
        # self.encoder.reset_classifier(out_size)
        return self.encoder

    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict

    def encoding_mask_noise(self, features, mask_rate=0.3):
        num_nodes = features.shape[0]
        perm = torch.randperm(num_nodes, device=features.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_features = features.clone()
        token_nodes = mask_nodes
        out_features[mask_nodes] = 0.0

        out_features[token_nodes] += self.enc_mask_token

        return out_features, (mask_nodes, keep_nodes)

    def random_remask(self, rep, remask_rate=0.5):

        num_nodes = rep.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep