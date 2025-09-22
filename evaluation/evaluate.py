import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(multi_class='multinomial', max_iter=1000)

    @torch.no_grad()
    def extract_embeddings(self, model, batches):
        """
        从批量数据中提取所有样本的嵌入
        :param model: 训练好的图神经网络模型
        :param batches: 数据批量对象
        :return: (embeddings, labels) 的元组
        """
        model.eval()
        all_embs, all_labels = [], []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        for batch in tqdm(batches, desc="Extracting embeddings"):
            batch = batch.to(self.device)
            embs = model.inference(batch)
            all_embs.append(embs)
            all_labels.append(batch.y)

        all_embs = torch.cat(all_embs, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        return all_embs, all_labels

    def train_classifier(self, model, train_batches):
        """
        :param model: 训练好的图神经网络模型
        :param train_batches: 训练数据批量对象
        """
        train_embs, train_labels = self.extract_embeddings(model, train_batches)
        train_embs = self.scaler.fit_transform(train_embs)  # 归一化特征
        self.clf.fit(train_embs, train_labels)

    def evaluate_embeddings(self, model_mame, model, test_batches, evaluator):
        """
        评估嵌入质量
        :param model: 训练好的图神经网络模型
        :param test_batches: 测试数据批量对象
        :param evaluator: 评估函数
        :return: 评估指标字典
        """
        if model_mame in ["gcn","gin","gat"]:
            output, test_labels = self.extract_embeddings(model, test_batches)
            output = output.argmax(axis=1)
            return evaluator(torch.tensor(test_labels), torch.tensor(output))
        else:
            test_embs, test_labels = self.extract_embeddings(model, test_batches)
            test_embs = self.scaler.transform(test_embs)
            y_pred = self.clf.predict(test_embs)

            if test_labels.shape != y_pred.shape:
                raise ValueError(f"Mismatch in label shape: {test_labels.shape} vs {y_pred.shape}")

            return evaluator(torch.tensor(test_labels).view(-1,1), torch.tensor(y_pred).view(-1,1))

    def visualize_embeddings(self, embeddings, labels, method="pca"):
        """
        可视化嵌入空间（使用 PCA 或 t-SNE 降维）
        :param embeddings: 嵌入矩阵 [n_samples, emb_dim]
        :param labels: 对应标签 [n_samples]
        :param method: 降维方法 ('pca' 或 'tsne')
        """
        if method == "pca":
            reducer = PCA(n_components=2)
            title = "PCA Embedding Visualization"
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
            title = "t-SNE Embedding Visualization"
        else:
            raise ValueError("method should be 'pca' or 'tsne'")

        embeddings_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()



# 使用示例 ---------------------------------------------------



    # 步骤2：评估嵌入质量
    # metrics = evaluator.evaluate_embeddings(embeddings, labels)
    # print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    # print("\nClassification Report:")
    # print(classification_report(None, None,
    #                             target_names=[str(i) for i in range(10)],  # 替换为你的类别名称
    #                             output_dict=False,
    #                             digits=4))

    # 步骤3（可选）：可视化嵌入空间
    # evaluator.visualize_embeddings(embeddings, labels)

# def evaluate_model(val_batchs, model, device, name=""):
#     model.eval()
#     output = model
#
#     for batch, _ in tqdm(eval_loader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         emb = model.emb(batch)
#         output.append(emb.cpu())
#     output = torch.cat(output, 0)
#
#     if args.incontext_eval:
#         acc = incontext_evaluate(args, output, name)
#         graph = eval_tag.graph
#         graph.ndata["feat"] = output
#         output, cat_output = model.inference(graph, device, args.eval_batch_size)
#         acc = incontext_evaluate(args, output, name)
#         acc = incontext_evaluate(args, cat_output, name)
#
#     if name in ['cora', 'pubmed', 'arxiv', 'products', 'wikics']:
#         graph = eval_tag.graph
#         test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
#         wandb.log({
#             f"{name}_estp_test_acc_lm": estp_test_acc,
#             f"{name}_best_val_acc_lm": best_val_acc,
#         })
#
#         if args.gnn_type != "":
#             graph.ndata["feat"] = output
#             output, cat_output = model.inference(graph, device, args.eval_batch_size)
#             test_acc, estp_test_acc, best_val_acc = node_classification_evaluation(graph, output, eval_tag.labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
#             wandb.log({
#                 f"{name}_estp_test_acc_gnn": estp_test_acc,
#                 f"{name}_best_val_acc_gnn": best_val_acc,
#             })
#
#     elif name in ['FB15K237', 'WN18RR']:
#         graph = eval_tag.graph
#         node_pairs = torch.LongTensor(eval_tag.test_graph["train"][0] + eval_tag.test_graph["valid"][0] + eval_tag.test_graph["test"][0])
#         labels = torch.LongTensor(eval_tag.test_graph["train"][1] + eval_tag.test_graph["valid"][1] + eval_tag.test_graph["test"][1])
#         test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
#         wandb.log({
#             f"{name}_estp_test_acc_lm": estp_test_acc,
#             f"{name}_best_val_acc_lm": best_val_acc,
#         })
#         if args.gnn_type != "":
#             graph.ndata["feat"] = output
#             output, cat_output = model.inference(graph, device, args.eval_batch_size)
#             test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
#             wandb.log({
#                 f"{name}_estp_test_acc_gnn": estp_test_acc,
#                 f"{name}_best_val_acc_gnn": best_val_acc,
#             })
#             test_acc, estp_test_acc, best_val_acc = edge_classification_evaluation(graph, cat_output, node_pairs, labels, eval_tag.split_idx, eval_tag.data_info["n_labels"], args.lr_f, args.weight_decay_f, args.lp_epochs, device)
#             wandb.log({
#                 f"{name}_estp_test_acc_gnn_cat": estp_test_acc,
#                 f"{name}_best_val_acc_gnn_cat": best_val_acc,
#             })
#     else:
#         graph = eval_tag.test_graph
#         link_prediction_evaluation(graph, output)
#
#     return output
#
# def evaluate_mol(args, model, device, name=""):
#     eval_mol = Mol(args, name)
#     dataset = IterMolDataset(eval_mol, 0, args.batch_size)
#     eval_loader = DataLoader(dataset, shuffle=False, batch_size=None)
#     model.eval()
#     pooler = pool(args.pooler)
#     output_lm, output_gnn, output_gnn_cat = [], [], []
#     labels = []
#
#     if args.incontext_eval:
#         acc = incontext_evaluate(args, None, output_lm, name)
#         acc = incontext_evaluate(args, None, output_gnn, name)
#         acc = incontext_evaluate(args, None, output_gnn_cat, name)
#
#     evaluator = Evaluator(name='ogbg-molhiv' if name == "hiv" else 'ogbg-molpcba' if name == "pcba" else 'ogbg-molchembl')
#     test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_lm, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
#     wandb.log({
#         "estp_test_acc_lm": estp_test_acc,
#         "best_val_acc_lm": best_val_acc,
#     })
#     if args.gnn_type != "":
#         test_acc, estp_test_acc, best_val_acc = graph_classification_evaluation(output_gnn, eval_mol.labels, eval_mol.split_idx, eval_mol.num_tasks, args.lr_f, args.weight_decay_f, args.lp_epochs, evaluator, device)
#         if best_val_acc > evaluate_mol.g_val_acc_gnn:
#             evaluate_mol.g_val_acc_gnn = best_val_acc
#             evaluate_mol.g_test_acc_gnn = estp_test_acc
#         wandb.log({
#             "estp_test_acc_gnn": estp_test_acc,
#             "best_val_acc_gnn": best_val_acc,
#         })
#



# def node_classification_evaluation(graph, x, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
#                                    mute=False):
#     in_feat = x.shape[1]
#     encoder = LogisticRegression(in_feat, num_classes)
#
#     num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
#     if not mute:
#         print(f"num parameters for finetuning: {sum(num_finetune_params)}")
#
#     encoder.to(device)
#     optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
#     # num_nodes = graph.num_nodes()
#     final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x.shape[0],
#                                                                                             x, labels, split_idx,
#                                                                                             optimizer_f, max_epoch_f,
#                                                                                             device, mute)
#     return final_acc, estp_acc, best_val_acc
#
#
# def edge_classification_evaluation(graph, x, node_pairs, labels, split_idx, num_classes, lr_f, weight_decay_f,
#                                    max_epoch_f, device, mute=False):
#     in_feat = x.shape[1] * 2
#     encoder = LogisticRegression(in_feat, num_classes)
#
#     # labels = torch.cat([labels['train'], labels['valid'], labels['test']], dim=0)
#     # train_cat_x = torch.cat([x[node_pairs['train'][:, 0]], x[node_pairs['train'][:, 1]]], dim=1)
#     # valid_cat_x = torch.cat([x[node_pairs['valid'][:, 0]], x[node_pairs['valid'][:, 1]]], dim=1)
#     # test_cat_x = torch.cat([x[node_pairs['test'][:, 0]], x[node_pairs['test'][:, 1]]], dim=1)
#     # x = torch.cat([train_cat_x, valid_cat_x, test_cat_x], dim=0)
#     x = torch.cat([x[node_pairs[:, 0]], x[node_pairs[:, 1]]], dim=1)
#     # split_idx = {'train': torch.arange(0, train_cat_x.shape[0]), 'valid': torch.arange(train_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]), 'test': torch.arange(train_cat_x.shape[0]+valid_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]+test_cat_x.shape[0])}
#     print(f"split_idx: {split_idx['train'].shape}, {split_idx['valid'].shape}, {split_idx['test'].shape}")
#
#     num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
#     if not mute:
#         print(f"num parameters for finetuning: {sum(num_finetune_params)}")
#
#     encoder.to(device)
#     optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
#     final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x.shape[0],
#                                                                                             x, labels, split_idx,
#                                                                                             optimizer_f, max_epoch_f,
#                                                                                             device, mute)
#     return final_acc, estp_acc, best_val_acc
#
#
# def link_prediction_evaluation(graph, x, node_pairs, labels, split_idx, num_classes, lr_f, weight_decay_f,
#                                max_epoch_f, device, mute=False):
#     in_feat = x.shape[1] * 2
#     encoder = LogisticRegression(in_feat, num_classes)
#
#     # labels = torch.cat([labels['train'], labels['valid'], labels['test']], dim=0)
#     # train_cat_x = torch.cat([x[node_pairs['train'][:, 0]], x[node_pairs['train'][:, 1]]], dim=1)
#     # valid_cat_x = torch.cat([x[node_pairs['valid'][:, 0]], x[node_pairs['valid'][:, 1]]], dim=1)
#     # test_cat_x = torch.cat([x[node_pairs['test'][:, 0]], x[node_pairs['test'][:, 1]]], dim=1)
#     # x = torch.cat([train_cat_x, valid_cat_x, test_cat_x], dim=0)
#     x = torch.cat([x[node_pairs[:, 0]], x[node_pairs[:, 1]]], dim=1)
#     # split_idx = {'train': torch.arange(0, train_cat_x.shape[0]), 'valid': torch.arange(train_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]), 'test': torch.arange(train_cat_x.shape[0]+valid_cat_x.shape[0], train_cat_x.shape[0]+valid_cat_x.shape[0]+test_cat_x.shape[0])}
#     print(f"split_idx: {split_idx['train'].shape}, {split_idx['valid'].shape}, {split_idx['test'].shape}")
#
#     num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
#     if not mute:
#         print(f"num parameters for finetuning: {sum(num_finetune_params)}")
#
#     encoder.to(device)
#     optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
#     final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x.shape[0],
#                                                                                             x, labels, split_idx,
#                                                                                             optimizer_f, max_epoch_f,
#                                                                                             device, mute)
#     return final_acc, estp_acc, best_val_acc
#
#
# def graph_classification_evaluation(x, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, evaluator,
#                                     device, mute=False):
#     in_feat = x.shape[1]
#     encoder = LogisticRegression(in_feat, num_classes)
#
#     num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
#     if not mute:
#         print(f"num parameters for finetuning: {sum(num_finetune_params)}")
#
#     encoder.to(device)
#     optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
#     # num_nodes = graph.num_nodes()
#     final_acc, estp_acc, best_val_acc = linear_probing_for_graph_classification(encoder, x.shape[0], x, labels,
#                                                                                 split_idx, optimizer_f, max_epoch_f,
#                                                                                 evaluator, device, mute)
#     return final_acc, estp_acc, best_val_acc
#
#
# def linear_probing_for_transductive_node_classification(model, graph, num_nodes, feat, labels, split_idx, optimizer,
#                                                         max_epoch, device, mute=False):
#     criterion = torch.nn.CrossEntropyLoss()
#
#     graph = graph.to(device)
#     x = feat.to(device)
#     labels = labels.to(device)
#
#     train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#     if not torch.is_tensor(train_idx):
#         train_idx = torch.as_tensor(train_idx)
#         val_idx = torch.as_tensor(val_idx)
#         test_idx = torch.as_tensor(test_idx)
#
#     # num_nodes = graph.num_nodes()
#     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
#     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
#     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)
#     # train_mask = graph.ndata["train_mask"]
#     # val_mask = graph.ndata["val_mask"]
#     # test_mask = graph.ndata["test_mask"]
#     # labels = graph.ndata["label"]
#
#     best_val_acc = 0
#     best_val_epoch = 0
#     best_model = None
#
#     if not mute:
#         epoch_iter = tqdm(range(max_epoch))
#     else:
#         epoch_iter = range(max_epoch)
#
#     for epoch in epoch_iter:
#         model.train()
#         out = model(graph, x)
#         loss = criterion(out[train_mask], labels[train_mask])
#         optimizer.zero_grad()
#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
#         optimizer.step()
#
#         with torch.no_grad():
#             model.eval()
#             pred = model(graph, x)
#             val_acc = accuracy(pred[val_mask], labels[val_mask])
#             val_loss = criterion(pred[val_mask], labels[val_mask])
#             test_acc = accuracy(pred[test_mask], labels[test_mask])
#             test_loss = criterion(pred[test_mask], labels[test_mask])
#
#         if val_acc >= best_val_acc:
#             best_val_acc = val_acc
#             best_val_epoch = epoch
#             best_model = copy.deepcopy(model)
#
#         if not mute:
#             epoch_iter.set_description(
#                 f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")
#
#     best_model.eval()
#     with torch.no_grad():
#         pred = best_model(graph, x)
#         estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
#     if mute:
#         print(
#             f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
#     else:
#         print(
#             f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
#
#     # (final_acc, es_acc, best_acc)
#     return test_acc, estp_test_acc, best_val_acc
#
#
# def linear_probing_for_graph_classification(model, num_nodes, feat, labels, split_idx, optimizer, max_epoch, evaluator,
#                                             device, mute=False):
#     criterion = torch.nn.BCEWithLogitsLoss()
#
#     # graph = graph.to(device)
#     x = feat.to(device)
#     labels = labels.to(device)
#     # model = model.to(device)
#
#     train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#     if not torch.is_tensor(train_idx):
#         train_idx = torch.as_tensor(train_idx)
#         val_idx = torch.as_tensor(val_idx)
#         test_idx = torch.as_tensor(test_idx)
#     print(train_idx.shape, val_idx.shape, test_idx.shape)
#
#     # num_nodes = graph.num_nodes()
#     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
#     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
#     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)
#     # train_mask = graph.ndata["train_mask"]
#     # val_mask = graph.ndata["val_mask"]
#     # test_mask = graph.ndata["test_mask"]
#     # labels = graph.ndata["label"]
#
#     best_val_acc = 0
#     best_val_epoch = 0
#     best_model = None
#
#     if not mute:
#         epoch_iter = tqdm(range(max_epoch))
#     else:
#         epoch_iter = range(max_epoch)
#
#     for epoch in epoch_iter:
#         model.train()
#         out = model(None, x)
#         ## ignore nan targets (unlabeled) when computing training loss.
#         is_labeled = labels[train_mask] == labels[train_mask]
#         # print(is_labeled)
#         # print(out[train_mask][is_labeled])
#         # print(torch.isnan(labels[train_mask][is_labeled]).any())
#         loss = criterion(out[train_mask][is_labeled], labels[train_mask][is_labeled])
#         optimizer.zero_grad()
#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
#         optimizer.step()
#
#         with torch.no_grad():
#             model.eval()
#             pred = model(None, x)
#             # print(labels[val_mask].shape)
#             # print(pred[val_mask].shape)
#             # print(evaluator.eval({'y_true': labels[val_mask], 'y_pred': pred[val_mask]}))
#
#             # print(pred[val_mask])
#             val_acc = list(evaluator.eval({'y_true': labels[val_mask], 'y_pred': pred[val_mask]}).values())[0]
#             # print(val_acc)
#             # (pred[val_mask], labels[val_mask])
#             is_labeled = labels[val_mask] == labels[val_mask]
#             val_loss = criterion(pred[val_mask][is_labeled], labels[val_mask][is_labeled])
#             test_acc = list(evaluator.eval({'y_true': labels[test_mask], 'y_pred': pred[test_mask]}).values())[0]
#             # accuracy(pred[test_mask], labels[test_mask])
#             is_labeled = labels[test_mask] == labels[test_mask]
#             test_loss = criterion(pred[test_mask][is_labeled], labels[test_mask][is_labeled])
#
#         if val_acc >= best_val_acc:
#             best_val_acc = val_acc
#             best_val_epoch = epoch
#             best_model = copy.deepcopy(model)
#
#         if not mute:
#             epoch_iter.set_description(
#                 f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_metric:{val_acc}, test_loss:{test_loss.item(): .4f}, test_metric:{test_acc: .4f}")
#
#     best_model.eval()
#     with torch.no_grad():
#         pred = best_model(None, x)
#         estp_test_acc = list(evaluator.eval({'y_true': labels[test_mask], 'y_pred': pred[test_mask]}).values())[0]
#         # accuracy(pred[test_mask], labels[test_mask])
#     if mute:
#         print(
#             f"# IGNORE: --- Testmetric: {test_acc:.4f}, early-stopping-Testmetric: {estp_test_acc:.4f}, Best Valmetric: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
#     else:
#         print(
#             f"--- Testmetric: {test_acc:.4f}, early-stopping-Testmetric: {estp_test_acc:.4f}, Best Valmetric: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
#
#     # (final_acc, es_acc, best_acc)
#     return test_acc, estp_test_acc, best_val_acc