from functools import partial
from typing import (
    Optional,
    Union,
    Callable,
)
import torch
from torchmetrics import Metric
from datasets_process.data import *
from datasets_process import *
from evaluation import *
from tasks import *
from dataset_info import *
from utils.graph import print_dataset_summary
from para import *
from models.text_encoder import LLMModel
from models.TAGFM_llm import TAGFM
from models.GNNs_llm import GNNs
from models.GraphCL_llm import GraphCL
from models.GraphMAE2_llm import GraphMAE2
# from models.TAGFM import TAGFM
from utils.functions import *
from utils.graph import  edge_dropout
from transformers import get_scheduler
from tqdm.auto import tqdm
import itertools
from models.BGRL_llm import BGRL



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
    return [get_task(name, task_type, split, root, transform, pre_transform, pre_filter, **kwargs) for name, task_type in
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

if __name__ == '__main__':

    args = build_args()
    task_save_name = f"{args.datasets_name}_{args.task_type}"
    eval_task_save_name = f"{args.eval_datasets}_{args.eval_task_type}"
    device, device_count = get_available_devices(args.device)

    log_file = setup_logger(args.datasets_name,args.eval_datasets)
    logging.info(f"Running on {device} ({device_count} devices available)")
    logging.info(f"Training configuration:\n{str(vars(args))}")
    logging.info(f"Log file: {log_file}")

    text_encoder = LLMModel(args.lm_name, args.mask_rate, device)
    train_tasks = get_tasks(
        args.datasets_name, args.task_type,
        split="train", save_data=True, load_saved=True,
        save_name=f"{task_save_name}_train", use_ppr_sampling=True
    )
    eval_tasks_train = get_tasks(
        args.eval_datasets, args.eval_task_type,
        split="train", save_data=True, load_saved=True,
        save_name=f"{eval_task_save_name}_train", use_ppr_sampling=True
    )
    eval_tasks_test = get_tasks(
        args.eval_datasets, args.eval_task_type,
        split="test", save_data=True, load_saved=True,
        save_name=f"{eval_task_save_name}_test", use_ppr_sampling=True
    )
    metric_names, evaluators = get_evaluators(args.eval_datasets, args.task_type)

    if args.model_name == "TAGLAS":
        model = TAGFM(args, text_encoder).to(device)
    elif args.model_name in ["gcn", "gin", "gat"]:
        model = GNNs(args, text_encoder, train_tasks[0].num_classes).to(device)
    elif args.model_name == "graphcl":
        model = GraphCL(args, text_encoder).to(device)
    elif args.model_name == "GraphMAE2":
        model = GraphMAE2(args, text_encoder).to(device)
    elif args.model_name == "bgrl":
        model = BGRL(args,text_encoder).to(device)

    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    num_training_steps = com_all_epoch(train_tasks, args)

    for task_idx, train_task in enumerate(train_tasks):
        logging.info("\n" + "=" * 40)
        logging.info(f"Processing task {task_idx + 1}/{len(train_tasks)}")
        logging.info(f"Dataset: {train_task.dataset_name}")

        train_batches, num_steps = dataloader(train_task, args.lm_name, text_encoder, args.batch_size, args.num_epochs)
        progress_bar = tqdm(total=args.num_epochs * len(train_batches),
                            desc=f"Task {task_idx + 1} Training")

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_batches):
                batch = batch.to(device)
                try:
                    if args.model_name in ["graphcl","bgrl"]:
                        edge_index1 = edge_dropout(batch.edge_index, 0.2)
                        edge_index2 = edge_dropout(batch.edge_index, 0.2)
                        loss, latent_loss = model(batch, edge_index1, edge_index2, epoch=epoch)
                    else:
                        loss, latent_loss = model(batch, epoch=epoch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    progress_bar.update(1)
                    mem_info = f"GPU{args.device}: {torch.cuda.memory_allocated() // 1024 ** 2}MB"
                    progress_bar.set_postfix({
                        "epoch": epoch + 1,
                        "loss": f"{loss.item():.4f}",
                        "memory":mem_info
                    })
                except Exception as e:
                    logging.error(f"Training failed at batch {batch_idx}: {str(e)}")
                    raise
            avg_loss = epoch_loss / len(train_batches)
            logging.info(f"Epoch {epoch + 1}/{args.num_epochs} - Avg Loss: {avg_loss:.4f}")
        final_model_path = save_model(model, args)
        progress_bar.close()

    logging.info("Starting evaluation...")
    for eval_task_idx, (eval_task_train, eval_task_test, metric_name, evaluator) in enumerate(
            zip(eval_tasks_train, eval_tasks_test, metric_names, evaluators)):
        eval_train_batches, _ = dataloader(eval_task_train, args.lm_name, text_encoder, args.eval_batch_size, args.num_epochs)
        eval_test_batches, _ = dataloader(eval_task_test, args.lm_name, text_encoder, args.eval_batch_size, args.num_epochs)
        logging.info(f"Processing task {eval_task_idx + 1}/{len(eval_tasks_train)}")
        logging.info(f"Dataset: {eval_task_train.dataset_name}")
        emb_evaluator = EmbeddingEvaluator()
        if args.model_name in ["TAGLAS","graphcl","GraphMAE2","bgrl"]:
            emb_evaluator.train_classifier(model, eval_train_batches)
        metrics = emb_evaluator.evaluate_embeddings(args.model_name,model, eval_test_batches, evaluator)
        logging.info(f"Evaluation metrics ({metric_name}): {metrics}")







