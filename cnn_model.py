import torch
import torch.nn as nn
import torch.nn.functional as F


class NIST_CNN(nn.Module):
    def __init__(self, save_path="./cnn_model_params.pth"):
        super(NIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, 10)
        self.loss = nn.CrossEntropyLoss()
        self.save_path = save_path

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc2(x)
        return x

    def extract_features(self, x):
        """
        提取特征的方法。
        - 如果输入是 Tensor (batch_size, 1, 28, 28)，返回特征 Tensor (batch_size, 50)。
        - 如果输入是 DataLoader，遍历提取所有数据特征，返回 (labels, features) 元组。
        """
        # Case 1: 输入是 DataLoader
        if isinstance(x, torch.utils.data.DataLoader):
            all_features = []
            all_labels = []
            device = next(self.parameters()).device

            was_training = self.training
            self.eval()

            with torch.no_grad():
                for batch in x:
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch[0], batch[1]
                    elif isinstance(batch, dict):
                        inputs = batch.get('x', batch.get('input'))
                        labels = batch.get('y', batch.get('label'))
                    else:
                        raise ValueError("DataLoader batch format not supported.")

                    inputs = inputs.to(device)
                    features = self.extract_features(inputs)

                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())

            self.train(was_training)

            return torch.cat(all_labels, dim=0), torch.cat(all_features, dim=0)

        # Case 2: 输入是 Tensor
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        return x

    def compute_grad_and_hessian(self, data_loader, init_theta=None):
        """
        【新想法核心】计算梯度 g 和 Hessian 矩阵 H。

        Args:
            data_loader: 客户端的数据加载器
            init_theta: 初始参数（可选，如果不提供则使用当前模型参数）

        Returns:
            g_mean: 梯度均值，形状 (C, D) = (10, 50)
            H_mean: Hessian 矩阵均值，形状 (C, C, D, D) = (10, 10, 50, 50)
        """
        device = next(self.parameters()).device
        self.eval()

        # 如果提供了 init_theta，先更新 fc2 的权重
        if init_theta is not None:
            init_theta = init_theta.to(device)
            with torch.no_grad():
                # init_theta 形状为 (10, 50)，与 fc2.weight 形状 (10, 50) 一致，无需转置
                self.fc2.weight.data = init_theta
                self.fc2.bias.data = torch.zeros(10).to(device)

        # 收集所有样本的梯度统计
        all_grads = []
        all_hessians = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('x', batch.get('input'))
                    labels = batch.get('y', batch.get('label'))
                else:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                # 提取特征
                features = self.extract_features(inputs)

                # 前向传播获取 logits
                logits = features @ self.fc2.weight.data.T + self.fc2.bias.data
                probs = torch.softmax(logits, dim=1)

                # One-hot 编码
                C = 10
                D = features.shape[1]

                Y_one_hot = F.one_hot(labels.long(), num_classes=C).float()
                diff = probs - Y_one_hot

                # 计算每个样本的梯度: (N, C, D)
                grad_per_sample = diff.unsqueeze(2) * features.unsqueeze(1)
                all_grads.append(grad_per_sample)

                # 计算 Hessian 矩阵
                P_unsqueeze = probs.unsqueeze(2)
                P_term = torch.diag_embed(probs) - (P_unsqueeze @ P_unsqueeze.transpose(1, 2))

                X_unsqueeze = features.unsqueeze(2)
                X_term = X_unsqueeze @ X_unsqueeze.transpose(1, 2)

                # Kronecker 积
                hessian_per_sample = P_term.unsqueeze(3).unsqueeze(4) * X_term.unsqueeze(1).unsqueeze(2)
                all_hessians.append(hessian_per_sample)

        # 合并所有 batch
        all_grads = torch.cat(all_grads, dim=0)
        all_hessians = torch.cat(all_hessians, dim=0)

        # 计算均值
        g_mean = all_grads.mean(dim=0)
        H_mean = all_hessians.mean(dim=0)

        return g_mean, H_mean
