import os
import argparse
import random
import yaml
import logging
from functools import partial
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from transformers import AutoTokenizer

import dgl

import torch
import torch.nn as nn
from torch import optim as optim

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def build_args():
    parser = argparse.ArgumentParser(description="unigraph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_entity", type=str, default="xxx")
    parser.add_argument("--datasets_name", nargs='+', type=str, default=["bbbp"],
                        choices=["cora_node","pubmed_node","arxiv_node","wikics","products",
                                "cora_link","pubmed_link","arxiv_link","fb15k237","wn18rr","ml1m",
                                "hiv","pcba","bbbp","bace","cyp450","muv","esol","freesolv","lipo",
                                "expla_graph","scene_graph","wiki_graph","ultrachat200k"])
    parser.add_argument("--eval_datasets", nargs='+', type=str, default=["bbbp"],
                        choices=["cora_node", "pubmed_node", "arxiv_node", "wikics", "products",
                                 "cora_link", "pubmed_link", "arxiv_link", "fb15k237", "wn18rr", "ml1m",
                                 "hiv", "pcba", "bbbp", "bace", "cyp450", "muv", "esol", "freesolv", "lipo",
                                 "expla_graph", "scene_graph", "wiki_graph", "ultrachat200k"])
    parser.add_argument("--task_type", type=str, default="default_text",
                        choices=["default","subgraph","default_text","subgraph_text","QA"])
    parser.add_argument("--eval_task_type", type=str, default="default_text",
                        choices=["default", "subgraph", "default_text", "subgraph_text", "QA"])
    parser.add_argument("--eval_datasets_name", nargs='+', type=str, default=['ogbn-papers100M', 'ogbn-products'])
    parser.add_argument("--model_name", type=str, default="gin",choices=["TAGLAS","gcn","gin","gat","bgrl","graphcl","GraphMAE2"])
    parser.add_argument("--lm_name", type=str, default="ST", choices=["llama2_7b", "e5", "BERT", "ST", "deberta"])
    parser.add_argument("--gnn_type", type=str, default="gcn",choices=["gat","gcn","gin","linear"])
    parser.add_argument("--device", type=int, default=1)

    parser.add_argument("--hidden_size", type=int, default=768)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight decay")
    parser.add_argument("--lr_f", type=float, default=0.01)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--negative_slope", type=float, default=0.1, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training')

    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--cut_off", type=int, default=64)
    parser.add_argument("--num_roots", type=int, default=100)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--process_mode", type=str, default="TA")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    # parser.add_argument("--loss_type", type=str, default="sce")

    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lp_epochs", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=5)

    parser.add_argument("--pooler", type=str, default="mean")

    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)

    parser.add_argument("--incontext_eval", action="store_true", default=False)
    parser.add_argument("--eval_num_label", type=int, default=5)
    parser.add_argument("--eval_num_support", type=int, default=3)
    parser.add_argument("--eval_num_query", type=int, default=3)

    parser.add_argument("--grad_accum_steps", type=int, default=4)

    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()
    return args
