import torch
from typing import List, Dict

def fedavg(gradients_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """FedAvg: 普通联邦平均 (无防御)"""
    if not gradients_list:
        return {}
    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        aggregated_gradients[name] = torch.stack(grads).mean(dim=0)
    return aggregated_gradients


def coordinate_wise_median(gradients_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    坐标级中位数聚合 (Coordinate-wise Median)。
    对于每个参数位置，取所有客户端梯度对应位置的中位数。
    """
    if not gradients_list:
        return {}

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}

    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        stacked_grads = torch.stack(grads, dim=0)  # (num_clients, *param_shape)

        # 计算坐标级中位数
        median_val = torch.median(stacked_grads, dim=0).values

        aggregated_gradients[name] = median_val

    return aggregated_gradients


def coordinate_wise_trimmed_mean(gradients_list: List[Dict], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """Definition 2: Coordinate-wise trimmed mean"""
    if not gradients_list:
        return {}

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    m = len(gradients_list)
    k = int(m * beta)

    if 2 * k >= m:
        raise ValueError(f"Beta {beta} is too large for the number of clients {m}.")

    for name in param_names:
        grads = [client_grad['gradients'][name] for client_grad in gradients_list]
        stacked_grads = torch.stack(grads, dim=0)

        sorted_grads, _ = torch.sort(stacked_grads, dim=0)

        if k > 0:
            trimmed_grads = sorted_grads[k : m - k]
        else:
            trimmed_grads = sorted_grads

        mean_val = torch.mean(trimmed_grads, dim=0)
        aggregated_gradients[name] = mean_val

    return aggregated_gradients


def _compute_distances(grads_dict: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """计算所有梯度之间的欧氏距离矩阵"""
    ids = list(grads_dict.keys())
    m = len(ids)

    flat_grads = []
    for id_ in ids:
        grad_vec = torch.cat([grads_dict[id_][name].flatten() for name in grads_dict[id_].keys()])
        flat_grads.append(grad_vec)

    distances = torch.zeros(m, m)
    for i in range(m):
        for j in range(m):
            if i != j:
                distances[i, j] = torch.norm(flat_grads[i] - flat_grads[j])

    return distances


def multi_krum(gradients_list: List[Dict], n_remove: int = 4, n_select: int = 5) -> Dict[str, torch.Tensor]:
    """
    修正后的 Multi-Krum 聚合方法。
    算法逻辑：
    1. 为每个客户端计算其到最近的 (m - n_remove - 2) 个邻居的距离之和作为得分。
    2. 选取得分最低的前 n_select 个客户端。
    3. 将这 n_select 个客户端的梯度求平均。
    """
    if not gradients_list:
        return {}

    m = len(gradients_list)
    num_neighbors = max(1, m - n_remove - 2)
    n_select = min(n_select, m)

    grads_dict = {g['id']: g['gradients'] for g in gradients_list}
    distances = _compute_distances(grads_dict)

    scores = []
    for i in range(m):
        sorted_dists, _ = torch.sort(distances[i])
        score = torch.sum(sorted_dists[:num_neighbors+1]).item()
        scores.append(score)

    scores_tensor = torch.tensor(scores)
    _, selected_indices = torch.topk(scores_tensor, n_select, largest=False)

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}
    for name in param_names:
        selected_grads = [gradients_list[i]['gradients'][name] for i in selected_indices]
        aggregated_gradients[name] = torch.stack(selected_grads).mean(dim=0)

    return aggregated_gradients


def clustering_based(gradients_list: List[Dict], n_clusters: int = 2) -> Dict[str, torch.Tensor]:
    """基于聚类的聚合方法 (简单距离基准)"""
    if not gradients_list:
        return {}

    m = len(gradients_list)
    if m <= 2:
        return fedavg(gradients_list)

    grads_dict = {g['id']: g['gradients'] for g in gradients_list}
    distances = _compute_distances(grads_dict)

    # 计算每个节点到其他所有节点的总距离
    scores = distances.sum(dim=1)
    median_score = scores.median()

    # 认为距离中心过于遥远的节点是异常的，保留总距离 <= 中位数的节点
    selected_indices = (scores <= median_score).nonzero(as_tuple=True)[0]

    if len(selected_indices) == 0:
        return fedavg(gradients_list)

    param_names = gradients_list[0]['gradients'].keys()
    aggregated_gradients = {}

    for name in param_names:
        selected_grads = [gradients_list[i]['gradients'][name] for i in selected_indices]
        aggregated_gradients[name] = torch.stack(selected_grads).mean(dim=0)

    return aggregated_gradients
