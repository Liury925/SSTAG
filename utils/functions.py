import torch.nn.functional as F
import math
from datetime import datetime
from typing import (
    Optional,
    Union,
    Callable,
)
import torch
from torchmetrics import Metric
from datasets_process.data import *
from evaluation import *
from dataset_info import *
from para import *
import copy

def get_available_devices(dev):
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[dev]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def dataloader(task,lm_name,text_encoder, batch_size, num_epochs=0):
    if hasattr(task, 'convert_text_to_embedding') and callable(getattr(task, 'convert_text_to_embedding')):
        task.convert_text_to_embedding(lm_name, text_encoder)
    num_batches = math.ceil(len(task) / batch_size)
    batchs =[]
    num_training_steps = num_batches * num_epochs
    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, len(task))
        batch = task.collate([task[i] for i in range(start, end)])
        batchs.append(batch)
    return batchs, num_training_steps


def com_all_epoch(tasks,args):
    num = 0
    for task in tasks:
        num_batches = math.ceil(len(task) / args.batch_size)
        num = num + num_batches
    num_training_steps = num * args.num_epochs
    return num_training_steps


def setup_logger(train_name,test_name):
    log_dir = f"./logs/{train_name}"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{test_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def save_model(model, args, epoch=None):
    """保存模型检查点"""
    save_dir = f"./saved_models/{args.model_name}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{args.datasets_name}_epoch{epoch}.pt" if epoch else f"{args.datasets_name}_final.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")
    return save_path

def load_model(model, args, epoch=None, device="cpu"):
    load_dir = f"./saved_models/{args.model_name}"
    filename = f"{args.datasets_name}_epoch{epoch}.pt" if epoch else f"{args.datasets_name}_final.pt"
    load_path = os.path.join(load_dir, filename)

    if not os.path.exists(load_path):
        logging.error(f"Model checkpoint not found: {load_path}")
        return None

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    logging.info(f"Model loaded from {load_path}")
    return model



def get_dataset(
        name: str,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs) -> TAGDataset:
    dataset = DATASET_TO_CLASS_DICT[DATASET_INFOR_DICT[name]["dataset"]](root=root, transform=transform,
                                                                             pre_transform=pre_transform,
                                                                             pre_filter=pre_filter, **kwargs)
    return dataset


def get_datasets(names: Union[str, list[str]],
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs) -> list[TAGDataset]:
    if isinstance(names, str):
        return [get_dataset(names, root, transform, pre_transform, pre_filter, **kwargs)]
    else:
        return [get_dataset(name, root, transform, pre_transform, pre_filter, **kwargs) for name in names]


def get_task(
        name: str,
        task_type: str = "default",
        split: str = "train",
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs) -> BaseTask:
    dataset = get_dataset(name, root, transform, pre_transform, pre_filter, **kwargs)
    if task_type not in DATASET_INFOR_DICT[name]["task"].keys():
        avaliable_tasks = ', '.join(list(DATASET_INFOR_DICT[name]["task"].keys()))
        raise ValueError(f"The task type {task_type} is not supported for dataset {name}. "
                         f"The supported task types are {avaliable_tasks}")
    return DATASET_INFOR_DICT[name]["task"][task_type](dataset, split, **kwargs)


def get_tasks(names: Union[str, list[str]],
              task_types: Union[str, list[str]] = "default",
              root: Optional[str] = None,
              split: str = "train",
              transform: Optional[Callable] = None,
              pre_transform: Optional[Callable] = None,
              pre_filter: Optional[Callable] = None,
              **kwargs):
    if isinstance(names, str):
        names = [names]
    if isinstance(task_types, str):
        task_types = [task_types] * len(names)
    assert len(names) == len(task_types)
    return [get_task(name, task_type, split, root, transform, pre_transform, pre_filter, **kwargs) for name, task_type
            in
            zip(names, task_types)]


def get_evaluator(name: str,
                  task_type: str = "default") -> tuple[str, Metric]:
    task_type = "QA" if task_type == "QA" else "default"
    if task_type not in DATASET_INFOR_DICT[name]["evaluation"].keys():
        avaliable_evaluation = ', '.join(list(DATASET_INFOR_DICT[name]["evaluation"].keys()))
        raise ValueError(f"The evaluation of task type {task_type} is not supported for dataset {name}. "
                         f"The supported task types are {avaliable_evaluation}")
    metric_name, evaluator_args = DATASET_INFOR_DICT[name]["evaluation"][task_type]
    return metric_name, Evaluator(**evaluator_args)


def get_evaluators(names: Union[str, list[str]], task_types: Union[str, list[str]] = "default") \
        -> tuple[list[str], list[Metric]]:
    if isinstance(names, str):
        names = [names]
    if isinstance(task_types, str):
        task_types = [task_types] * len(names)

    metric_names = []
    evaluator_list = []
    for name, task_type in zip(names, task_types):
        metric_name, evaluator_func = get_evaluator(name, task_type)
        metric_names.append(metric_name)
        evaluator_list.append(evaluator_func)
    return metric_names, evaluator_list