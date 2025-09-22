import json
import os
import os.path as osp
import shutil
from typing import (
    Optional,
    Callable,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from datasets_process import HF_REPO_ID
from datasets_process.data import TAGDataset, TAGData, BaseDict
from utils.graph import safe_to_undirected
from utils.io import download_url, extract_zip, move_files_in_dir, download_hf_file
from utils.dataset_split import generate_link_split_loop



class Arxiv(TAGDataset):
    r"""Arxiv citation network dataset.
    """
    zip_url = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
    node_text_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
    graph_description = "This is a citation network from Arxiv platform focusing on the computer science area. Nodes represent academic papers and edges represent citation relationships. "

    def __init__(self,
                 name: str = "arxiv",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        self.side_data.link_split, self.side_data.keep_edges = generate_link_split_loop(self._data.edge_index)

    def raw_file_names(self) -> list:
        return ["nodeidx2paperid.csv.gz", "labelidx2arxivcategeory.csv.gz", "edge.csv.gz",
                "node_year.csv.gz", "node-feat.csv.gz", "node-label.csv.gz", "train.csv.gz", "valid.csv.gz",
                "test.csv.gz", "titleabs.tsv", "arxiv.json"]

    def download(self) -> None:
        print("Downloading node text file...")
        _ = download_url(self.node_text_url, self.raw_dir)
        print("Downloading arxiv.json...")
        download_hf_file(HF_REPO_ID, subfolder="arxiv", filename="arxiv.json", cache_dir=self.raw_dir,  local_dir=self.raw_dir)
        print("Downloading and extracting dataset...")
        path = download_url(self.zip_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        dir_name = osp.join(self.raw_dir, "arxiv")
        move_files_in_dir(osp.join(dir_name, "raw"), self.raw_dir)
        move_files_in_dir(osp.join(dir_name, "mapping"), self.raw_dir)
        move_files_in_dir(osp.join(dir_name, "split/time"), self.raw_dir)
        shutil.rmtree(dir_name)

    def gen_data(self) -> tuple[list[TAGData], None]:
        print("Loading and processing data...")

        edge = pd.read_csv(self.raw_paths[2], compression='gzip', header=None).values.T.astype(np.int64)
        node_feat = pd.read_csv(self.raw_paths[4], compression='gzip', header=None).values.astype(np.float32)
        node_year = pd.read_csv(self.raw_paths[3], compression='gzip', header=None).values

        nodeidx2paperid = pd.read_csv(self.raw_paths[0], index_col="node idx")
        titleabs = pd.read_csv(self.raw_paths[-2], sep="\t", names=["paper id", "title", "abstract"],
                               index_col="paper id")

        # 处理 NaN 值，确保拼接时不出错
        titleabs = nodeidx2paperid.join(titleabs, on="paper id").fillna("")
        node_text_lst = ("Academic paper. Title: " + titleabs["title"] + ". Abstract: " + titleabs["abstract"]).values

        label = pd.read_csv(self.raw_paths[5], compression='gzip', header=None).values.squeeze()

        train_idx = torch.from_numpy(pd.read_csv(self.raw_paths[6], compression='gzip', header=None).values.T[0]).long()
        valid_idx = torch.from_numpy(pd.read_csv(self.raw_paths[7], compression='gzip', header=None).values.T[0]).long()
        test_idx = torch.from_numpy(pd.read_csv(self.raw_paths[8], compression='gzip', header=None).values.T[0]).long()

        edge_index = torch.from_numpy(edge).long()
        edge_index, _ = safe_to_undirected(edge_index)
        edge_text_lst = ["The connected two papers have a citation relationship."] * edge_index.size(-1)
        edge_map = torch.zeros(edge_index.size(-1), dtype=torch.long)

        x_original = torch.from_numpy(node_feat).float()
        node_year = torch.from_numpy(node_year).long()
        label_map = torch.from_numpy(label).long()

        with open(self.raw_paths[-1]) as f:
            label_desc = json.load(f)
        ordered_desc = BaseDict({item["name"]: item["description"] for item in label_desc})
        label_names = list(ordered_desc.keys())

        data = TAGData(
            x=node_text_lst, node_map=torch.arange(len(node_text_lst), dtype=torch.long), edge_index=edge_index,
            edge_attr=edge_text_lst, edge_map=edge_map, x_original=x_original, label=label_names,
            label_map=label_map, node_year=node_year
        )

        side_data = BaseDict(node_split={"train": train_idx, "val": valid_idx, "test": test_idx},
                             label_description=ordered_desc)

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list[int]]:
        """Return sample labels and their corresponding index for node-level tasks."""
        indexs = self.side_data.node_split[split]
        labels = self.label_map[indexs]
        return indexs, labels, labels.tolist()

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list[int]]:
        """Return sample labels and their corresponding index for link-level tasks."""
        indexs, labels = self.side_data.link_split[split]
        label_map = labels + 40
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most likely paper category for the target arxiv paper?"]
        answer_list = [f"{self.label[l]}." for l in label_map]
        a_list, a_idxs = np.unique(answer_list, return_inverse=True)
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list.tolist()

    def get_LQA_list(self, label_map, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["Do these two papers have a citation relationship? Answer 'yes' or 'no'."]
        answer_list = [f"{self.label[l]}." for l in label_map]
        a_list, a_idxs = np.unique(answer_list, return_inverse=True)
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list.tolist()