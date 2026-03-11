import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from typing import List, Dict


def get_mnist_client_dataloaders(
    num_clients: int,
    batch_size: int,
    root: str = './data',
    download: bool = True
) -> List[Dict]:
    """
    加载 MNIST 数据集，将其随机均匀分成 num_clients 份，并返回每个 client 的 dataloader。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    total_len = len(train_dataset)

    base_len = total_len // num_clients
    remainder = total_len % num_clients

    lengths = []
    for i in range(num_clients):
        length = base_len + 1 if i < remainder else base_len
        lengths.append(length)

    subsets = random_split(train_dataset, lengths)

    client_data_list = []
    for client_id, subset in enumerate(subsets):
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_dict = {
            "id": client_id,
            "dataloader": loader
        }
        client_data_list.append(client_dict)

    print(f"Data process finished: Split MNIST ({total_len} samples) into {num_clients} clients.")
    return client_data_list


def get_proxy_dataloader(
    batch_size: int,
    root: str = './data',
    download: bool = True
) -> DataLoader:
    """
    【新想法】获取服务器端干净代理数据的 DataLoader。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)

    # 取前 1000 个样本作为代理数据
    proxy_size = 1000
    proxy_dataset = Subset(train_dataset, range(proxy_size))
    proxy_loader = DataLoader(proxy_dataset, batch_size=batch_size, shuffle=True)

    print(f"Proxy data loader created: {proxy_size} samples")
    return proxy_loader


def cnn_reform_func_with_gh(
    g_mean: torch.Tensor,
    H_mean: torch.Tensor,
    init_theta: torch.Tensor
) -> pd.DataFrame:
    """
    【新想法】使用预计算的梯度 g 和 Hessian H 矩阵进行重构。

    Args:
        g_mean: 梯度均值，形状 (C, D) = (10, 50)
        H_mean: Hessian 矩阵均值，形状 (C, C, D, D) = (10, 10, 50, 50)
        init_theta: 初始参数，形状 (10, 50)

    Returns:
        pd.DataFrame: 包含 Y_tlide 和 X_tlide 的 DataFrame
    """
    device = g_mean.device
    init_theta = init_theta.to(device)

    C, D = g_mean.shape  # C=10, D=50

    # 1. 将 H  reshape 为 (C*D, C*D)
    H_mean = H_mean.permute(0, 2, 1, 3).reshape(C*D, C*D)

    # 2. 将 g 展平为 (C*D,)
    g_mean_flat = g_mean.flatten()

    # 3. 将 init_theta 展平
    theta_flat = init_theta.T.flatten()

    # 4. 特征分解 (Eigen Decomposition)
    L, V = torch.linalg.eigh(H_mean)

    # 处理小特征值
    mask = L > 1e-16
    L_safe = torch.where(mask, L, torch.tensor(0., device=device))

    # 计算 H^(-1/2) 和 H^(1/2)
    L_sqrt = torch.sqrt(L_safe)
    L_inv_sqrt = torch.where(mask, 1.0 / L_sqrt, torch.tensor(0., device=device))

    sqrtH = V @ torch.diag(L_sqrt) @ V.T
    sqrtH_inv = V @ torch.diag(L_inv_sqrt) @ V.T

    # 5. 计算 X_tlide, Y_tlide
    X_tlide = -sqrtH
    Y_tlide = (sqrtH_inv @ g_mean_flat) - (sqrtH @ theta_flat)

    # 6. 构建 DataFrame
    Y_np = Y_tlide.detach().cpu().numpy()
    X_np = X_tlide.detach().cpu().numpy()

    data = np.column_stack((Y_np, X_np))
    columns = ["Y_tlide"] + [f"X_tlide_{i+1}" for i in range(X_np.shape[1])]

    return pd.DataFrame(data, columns=columns)


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    K = 10
    BATCH_SIZE = 64

    clients_data = get_mnist_client_dataloaders(num_clients=K, batch_size=BATCH_SIZE)
    print(f"生成了 {len(clients_data)} 个 client 的数据")

    proxy_loader = get_proxy_dataloader(batch_size=BATCH_SIZE)
    print(f"Proxy loader batches: {len(proxy_loader)}")
