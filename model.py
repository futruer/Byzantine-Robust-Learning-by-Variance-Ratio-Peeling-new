from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
import copy
import config


class Client:

    def __init__(self, model: nn.Module = None,
                       data: torch.utils.data.DataLoader = None,
                       reform_func_with_gh: callable = None,
                       id: int = None,
                       is_attacker: bool = False,
                       attack_type: str = None):
        self.model = model
        self.data = data
        self.reform_func_with_gh = reform_func_with_gh
        self.id = id
        self.is_normal = True
        self.is_attacker = is_attacker
        self.attack_type = attack_type

    def compute_g_and_H(self, init_theta: torch.Tensor) -> None:
        """
        【新想法】使用广播的基准模型计算梯度 g 和 Hessian 矩阵 H。

        这个方法直接使用模型的 fc2 层计算梯度和 Hessian，
        无需进行本地训练。
        """
        device = next(self.model.parameters()).device

        # 复制模型并设置为 eval 模式
        temp_model = copy.deepcopy(self.model).to(device)
        temp_model.eval()

        # 使用提供的数据加载器计算 g 和 H
        g_mean, H_mean = temp_model.compute_grad_and_hessian(self.data, init_theta)

        # 【核心修改：立即执行梯度攻击】
        # 在恶意节点上，计算完 g 后立即对 g 进行攻击
        # 这样攻击后的 g 会用于后续的重构数据计算和 Beta 求解
        if self.is_attacker and self.attack_type:
            if self.attack_type in ['sign_flipping', 'mixed']:
                g_mean = -g_mean
            elif self.attack_type == 'scaling':
                g_mean = g_mean * 10.0
            elif self.attack_type == 'gaussian':
                # 高斯攻击：均值0.1，方差0.1
                g_mean = g_mean + torch.randn_like(g_mean) * 0.1 + 0.1

        # 保存到客户端
        self.g_mean = g_mean.to(device)
        self.H_mean = H_mean.to(device)

        # 清理
        del temp_model

    def get_XY_tlide_with_gh(self, init_theta: torch.Tensor) -> None:
        """
        【新想法】使用预计算的 g 和 H 矩阵获取重构数据。
        """
        if not hasattr(self, 'g_mean') or not hasattr(self, 'H_mean'):
            raise ValueError("请先调用 compute_g_and_H 方法计算 g 和 H 矩阵")

        # 使用 g, H 和 init_theta 重构数据
        XY_tlide = self.reform_func_with_gh(self.g_mean, self.H_mean, init_theta)
        self.XY_tlide = XY_tlide

    def XY_tlide_least_square(self, init_theta: torch.Tensor) -> torch.Tensor:
        """
        【新想法】最小二乘求解系数 beta。
        使用 g, H 矩阵进行重构。
        """
        self.get_XY_tlide_with_gh(init_theta)

        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)

        # 添加截距项 (全1列)
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)

        # 计算最小二乘系数: β = (X^T * X)^(-1) * X^T * Y
        XTX = torch.matmul(X_tlide_with_intercept.T, X_tlide_with_intercept)
        XTY = torch.matmul(X_tlide_with_intercept.T, Y_tlide.unsqueeze(1))

        # 使用伪逆来避免奇异矩阵问题
        try:
            coefficients = torch.matmul(torch.linalg.pinv(XTX), XTY)
        except:
            coefficients = torch.linalg.solve(XTX, XTY)

        return coefficients.squeeze()

    def compute_mse_list(self, LS_coefficients: List[dict]) -> None:
        """
        对于传入的每一个最小二乘系数 beta_k，计算当前客户端数据下
        残差向量 Y_tlide - X_tlide^T beta_k 的 L2 范数，并绑定对应的 client id。
        """
        # 从 DataFrame 中提取 Y_tlide (第一列) 和 X_tlide (其余列)
        Y_tlide = torch.tensor(self.XY_tlide.iloc[:, 0].values, dtype=torch.float32)
        X_tlide = torch.tensor(self.XY_tlide.iloc[:, 1:].values, dtype=torch.float32)

        # 添加截距项
        ones = torch.ones(X_tlide.shape[0], 1, dtype=torch.float32)
        X_tlide_with_intercept = torch.cat([ones, X_tlide], dim=1)

        # 将所有 beta_k 堆叠成矩阵 B: (d+1, K)
        betas = []
        client_ids = []
        for coeff_dict in LS_coefficients:
            beta = coeff_dict["beta_k"]
            client_id = coeff_dict["id"]
            if beta.dim() > 1:
                beta = beta.view(-1)
            betas.append(beta)
            client_ids.append(client_id)
        B = torch.stack(betas, dim=1)

        # 预测值矩阵: (n, K)
        preds = X_tlide_with_intercept @ B

        # 残差矩阵: (n, K)
        residuals = Y_tlide.unsqueeze(1) - preds

        # 对每个 beta_k 计算 L2 范数
        l2_norms = torch.sqrt(torch.sum(residuals ** 2, dim=0))

        # 将 L2 范数与对应的 client id 绑定
        self.mse_list = []
        for i, client_id in enumerate(client_ids):
            self.mse_list.append({
                "id": client_id,
                "mse": l2_norms[i]
            })

    def compute_rest_mse_med(self, normal_client_ids: list[int]) -> torch.Tensor:
        """
        从 mse_list 中挑选出 normal_client_ids 对应的元素，
        提取这些元素的 mse 值，计算中位数并返回。
        """
        filtered_mse_values = []
        for mse_item in self.mse_list:
            if mse_item["id"] in normal_client_ids:
                filtered_mse_values.append(mse_item["mse"])

        if len(filtered_mse_values) == 0:
            raise ValueError(f"No matching mse values found for normal_client_ids: {normal_client_ids}")

        mse_tensor = torch.stack(filtered_mse_values)
        mse_med = torch.median(mse_tensor)

        return mse_med

    def train(self, local_epochs: int = 1, learning_rate: float = 0.1) -> dict:
        """
        使用当前 client 的数据进行本地训练 (FedAvg 模式)。
        返回包含 client id 和伪梯度 (初始权重 - 更新后权重) 的字典。
        """
        if self.model is None or self.data is None:
            raise ValueError("Model or Data is not set.")

        self.model.train()
        device = next(self.model.parameters()).device

        # 保存本轮初始权重
        original_weights = {name: param.data.clone() for name, param in self.model.named_parameters()}

        # 初始化优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        total_loss = 0.0
        num_batches = 0

        # 本地多 Epoch 迭代
        for epoch in range(local_epochs):
            for batch_data in self.data:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, labels = batch_data
                elif isinstance(batch_data, dict):
                    inputs = batch_data.get('input', batch_data.get('x', None))
                    labels = batch_data.get('label', batch_data.get('y', None))
                else:
                    inputs = batch_data
                    labels = None

                if inputs is not None:
                    inputs = inputs.to(device)
                if labels is not None:
                    labels = labels.to(device)

                optimizer.zero_grad()

                if labels is not None:
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels.long())
                else:
                    outputs = self.model(inputs)
                    loss = outputs.mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # 计算伪梯度 (Pseudo-gradients): 初始权重 - 更新后的权重
        pseudo_gradients = {}
        for name, param in self.model.named_parameters():
            pseudo_gradients[name] = original_weights[name] - param.data.clone()

        # 【核心修改：立即执行梯度攻击】
        # 在恶意节点上，计算完伪梯度后立即对伪梯度进行攻击
        # 这样攻击后的梯度会上交给协调层进行聚合
        if self.is_attacker and self.attack_type:
            if self.attack_type in ['sign_flipping', 'mixed']:
                for name in pseudo_gradients:
                    pseudo_gradients[name] = -pseudo_gradients[name]
            elif self.attack_type == 'scaling':
                for name in pseudo_gradients:
                    pseudo_gradients[name] = pseudo_gradients[name] * 10.0
            elif self.attack_type == 'gaussian':
                # 高斯攻击：均值0.1，方差0.1
                for name in pseudo_gradients:
                    pseudo_gradients[name] = pseudo_gradients[name] + torch.randn_like(pseudo_gradients[name]) * 0.1 + 0.1

        # 将模型恢复到初始状态
        self.model.load_state_dict(original_weights)

        return {
            "id": self.id,
            "gradients": pseudo_gradients,
            "loss": total_loss / num_batches if num_batches > 0 else 0.0
        }


class Orchestrator:

    def __init__(self, model: nn.Module, clients: list[Client],
                 proxy_data_loader: torch.utils.data.DataLoader = None,
                 gradient_attack_func: callable = None):
        """
        初始化 Orchestrator。

        Args:
            model: 全局模型
            clients: 客户端列表
            proxy_data_loader: 服务器端代理数据加载器（用于训练基准模型）
            gradient_attack_func: 梯度攻击函数
        """
        self.model = model
        self.clients = clients
        self.proxy_data_loader = proxy_data_loader
        self.gradient_attack_func = gradient_attack_func

        self.init_normal_client_ids()

        # 【新想法】核心流程：
        # 1. 使用代理数据训练基准模型（信任根）
        # 2. 广播基准模型到所有客户端
        # 3. 各客户端计算 g 和 H 矩阵
        # 4. 使用 VR Peeling 进行筛选

        # 步骤1 & 2: 训练并广播基准模型
        self.init_theta = self.train_proxy_model()

        # 步骤3: 各客户端使用基准模型计算 g 和 H
        self.broadcast_model_and_compute_gh()

        # 步骤4: 收集各客户端的重构数据并计算 MSE
        self.compute_init_client_mse_list()

    def train_proxy_model(self) -> torch.Tensor:
        """
        【新想法】使用服务器端代理数据训练基准模型。
        如果已存在保存的基准模型，则直接加载。
        """
        import os

        # 检查是否存在已保存的基准模型
        proxy_model_path = config.PROXY_MODEL_SAVE_PATH

        if os.path.exists(proxy_model_path):
            print(f"\n[Proxy Model] 发现已保存的基准模型，正在加载...")
            device = next(self.model.parameters()).device
            proxy_model = copy.deepcopy(self.model).to(device)
            proxy_model.load_state_dict(torch.load(proxy_model_path, map_location=device))
            proxy_model.eval()
            init_theta = proxy_model.fc2.weight.data.clone()
            print(f"[Proxy Model] 基准模型加载完成，init_theta shape: {init_theta.shape}")
        else:
            print("\n[Proxy Model] 开始训练服务器端基准模型...")

            device = next(self.model.parameters()).device
            proxy_model = copy.deepcopy(self.model).to(device)
            proxy_model.train()

            optimizer = torch.optim.SGD(proxy_model.parameters(), lr=config.PROXY_LR)

            # 使用代理数据进行训练
            for epoch in range(config.PROXY_EPOCHS):
                total_loss = 0.0
                num_batches = 0

                for batch_data in self.proxy_data_loader:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        inputs, labels = batch_data
                    else:
                        continue

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = proxy_model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"  [Proxy Model] Epoch {epoch+1}/{config.PROXY_EPOCHS}, Loss: {avg_loss:.6f}")

            # 获取基准模型的 fc2 权重作为 init_theta
            init_theta = proxy_model.fc2.weight.data.clone()
            print(f"[Proxy Model] 基准模型训练完成，init_theta shape: {init_theta.shape}")

            # 保存基准模型供后续使用
            torch.save(proxy_model.state_dict(), proxy_model_path)
            print(f"[Proxy Model] 基准模型已保存到: {proxy_model_path}")

        # 保存基准模型供后续使用
        self.proxy_model = proxy_model

        return init_theta

    def broadcast_model_and_compute_gh(self) -> None:
        """
        【新想法】将基准模型广播到所有客户端，各客户端计算 g 和 H 矩阵。
        """
        print("\n[Broadcast] 广播基准模型到所有客户端...")

        # 将基准模型广播到所有客户端
        for client in self.clients:
            client.model = copy.deepcopy(self.proxy_model)
            client.model.eval()  # 设置为 eval 模式

            # 各客户端计算 g 和 H 矩阵
            # print(f"  [Client {client.id}] 计算 g 和 H 矩阵...")
            client.compute_g_and_H(self.init_theta)

        print(f"[Broadcast] 所有客户端 g 和 H 计算完成")

    def init_normal_client_ids(self) -> None:
        self.normal_client_dict_list: list[dict] = []
        for client in self.clients:
            self.normal_client_dict_list.append({
                "id": client.id,
                "is_normal": client.is_normal
            })

    def collect_client_betas(self) -> None:
        """
        【新想法】收集所有客户端的 beta 系数。
        """
        print("\n[Collect Betas] 使用 g, H 方法收集各客户端 beta...")
        self.LS_coefficients: List[torch.Tensor] = []
        for client in self.clients:
            beta = client.XY_tlide_least_square(self.init_theta)
            self.LS_coefficients.append({
                "id": client.id,
                "beta_k": beta
            })
            # print(f"  Client {client.id}: beta shape = {beta.shape}")

    def compute_init_client_mse_list(self) -> None:
        """
        计算每个客户端相对于所有客户端 beta 的 MSE 列表。
        """
        self.collect_client_betas()
        for client in self.clients:
            client.compute_mse_list(self.LS_coefficients)

    def collect_client_mse_med(self) -> list[dict]:
        """
        将 normal_client_ids 发送到每个 Client，
        各个 Client 挑选出正常 client 对应的 mse_list 中的元素并取中位数。
        """
        normal_client_ids = []
        for client_dict in self.normal_client_dict_list:
            if client_dict["is_normal"]:
                normal_client_ids.append(client_dict["id"])

        result_list = []
        for client in self.clients:
            mse_med = client.compute_rest_mse_med(normal_client_ids)
            result_list.append({
                "id": client.id,
                "mse_med": mse_med
            })

        return result_list

    def update_normal_clients_by_gap(self, mse_med_list: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据收集到的所有 Client 返回的中位数，对这些中位数进行排序，
        排序后进行一阶差分，找到最大gap，并将最大gap位置及之后所有位置的
        client的is_normal属性改为false。

        返回更新前的方差、更新后的方差以及方差比。
        """
        # 记录更新前 is_normal=True 的 client 的 mse_med 值
        before_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    before_mse_values.append(item["mse_med"])
                    break

        # 计算更新前的方差
        if len(before_mse_values) > 1:
            before_mse_tensor = torch.stack(before_mse_values)
            before_variance = torch.var(before_mse_tensor, unbiased=False)
        elif len(before_mse_values) == 1:
            before_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            before_variance = torch.tensor(0.0, dtype=torch.float32)

        # 按 mse_med 值进行排序
        sorted_list = sorted(mse_med_list, key=lambda x: x["mse_med"].item())

        # 提取排序后的 mse_med 值用于计算差分
        mse_values = [item["mse_med"] for item in sorted_list]

        if len(mse_values) < 2:
            return before_variance, before_variance, torch.tensor(1.0, dtype=torch.float32)

        # ========== 原版：直接计算差分 ==========
        mse_values_list = [val.item() for val in mse_values]
        diffs = [mse_values_list[i+1] - mse_values_list[i] for i in range(len(mse_values_list) - 1)]

        # ========== 改进版1：对数转换后计算差分 ==========
        # 使用对数转换消除极端值的影响，使 Max Gap 能正确识别正常节点和异常节点之间的跨量级鸿沟
        # mse_values_list = [math.log(val.item() + 1e-8) for val in mse_values]
        # diffs = [mse_values_list[i+1] - mse_values_list[i] for i in range(len(mse_values_list) - 1)]

        # ========== 改进版2：自适应平滑的对数转换 ==========
        # 使用当前批次 MSE 的均值作为稳定器，兼顾小数值和大数值的场景
        # raw_mse_list = [val.item() for val in mse_values]
        # adaptive_shift = sum(raw_mse_list) / len(raw_mse_list) + 1e-8
        # mse_values_list = [math.log(val + adaptive_shift) for val in raw_mse_list]
        # diffs = [mse_values_list[i+1] - mse_values_list[i] for i in range(len(mse_values_list) - 1)]

        # ========== 调试输出：原始 MSE 值 ==========
        # print(f"\n[DEBUG] 原始 MSE 值排序（从小到大）:")
        # for i, item in enumerate(sorted_list):
        #     print(f"  [{i}] Client {item['id']}: MSE = {item['mse_med'].item():.6f}")

        # print(f"\n[DEBUG] 对数转换后 MSE 值:")
        # for i, val in enumerate(mse_values_list):
        #     print(f"  [{i}] log(MSE) = {val:.6f}")

        # print(f"\n[DEBUG] 差分值 (diffs):")
        # for i, d in enumerate(diffs):
        #     print(f"  [{i}] gap between {sorted_list[i]['id']} and {sorted_list[i+1]['id']}: {d:.6f}")

        # 找到最大gap的位置
        max_gap_idx = diffs.index(max(diffs))

        # 记录最大gap位置之后的client id
        abnormal_client_ids = []
        for i in range(max_gap_idx + 1, len(sorted_list)):
            abnormal_client_ids.append(sorted_list[i]["id"])

        # 根据这些id将对应client的is_normal属性改为false
        for client in self.clients:
            if client.id in abnormal_client_ids:
                client.is_normal = False

        # 更新 self.normal_client_dict_list
        self.init_normal_client_ids()

        # 记录更新后 is_normal=True 的 client 的 mse_med 值
        after_mse_values = []
        for item in mse_med_list:
            client_id = item["id"]
            for client in self.clients:
                if client.id == client_id and client.is_normal:
                    after_mse_values.append(item["mse_med"])
                    break

        # 计算更新后的方差
        if len(after_mse_values) > 1:
            after_mse_tensor = torch.stack(after_mse_values)
            after_variance = torch.var(after_mse_tensor, unbiased=False)
        elif len(after_mse_values) == 1:
            after_variance = torch.tensor(0.0, dtype=torch.float32)
        else:
            after_variance = torch.tensor(0.0, dtype=torch.float32)

        # 计算方差比
        if before_variance.item() != 0:
            variance_ratio = after_variance / before_variance
        else:
            variance_ratio = torch.tensor(0.0, dtype=torch.float32) if after_variance.item() == 0 else torch.tensor(float('inf'), dtype=torch.float32)

        return before_variance, after_variance, variance_ratio

    def _save_client_states(self) -> dict:
        """保存所有client的is_normal状态"""
        states = {}
        for client in self.clients:
            states[client.id] = client.is_normal
        return states

    def _restore_client_states(self, states: dict) -> None:
        """恢复所有client的is_normal状态"""
        for client in self.clients:
            if client.id in states:
                client.is_normal = states[client.id]
        self.init_normal_client_ids()

    def iterative_update_normal_clients(self) -> dict:
        """
        迭代更新normal clients，记录最低方差比的时刻，并将该时刻的状态作为最终状态。
        """
        # 确保从所有client.is_normal均为true开始
        for client in self.clients:
            client.is_normal = True
        self.init_normal_client_ids()

        total_clients = len(self.clients)
        min_variance_ratio = float('inf')

        # 初始状态全为正常
        best_state = self._save_client_states()
        best_normal_client_dict_list = [dict(d) for d in self.normal_client_dict_list]

        iteration = 0
        iteration_records = []

        while True:
            normal_count = sum(1 for client in self.clients if client.is_normal)
            if normal_count < total_clients / 2:
                break

            mse_med_list = self.collect_client_mse_med()
            if normal_count == 0:
                break

            state_before_update = self._save_client_states()

            before_var, after_var, variance_ratio = self.update_normal_clients_by_gap(mse_med_list)
            variance_ratio_value = variance_ratio.item()

            new_normal_count = sum(1 for client in self.clients if client.is_normal)

            # 【核心兜底机制 Majority Fallback】
            if new_normal_count < total_clients / 2:
                print(f"  [Fallback Triggered] 尝试剥离后仅剩 {new_normal_count}/{total_clients} 个节点。")
                print(f"  [Fallback Triggered] 判定系统未遭受严重数据投毒，拒绝过度剥离并终止筛选！")

                self._restore_client_states(state_before_update)
                break

            print(f"  Variance ratio: {variance_ratio_value:.6f} (before:{before_var:.6f}, after:{after_var:.6f})")

            updated_state = self._save_client_states()
            updated_normal_client_dict_list = [dict(d) for d in self.normal_client_dict_list]

            iteration_records.append({
                "iteration": iteration,
                "variance_ratio": variance_ratio_value,
                "normal_count": new_normal_count,
            })

            if variance_ratio_value < min_variance_ratio:
                min_variance_ratio = variance_ratio_value
                best_state = updated_state.copy()
                best_normal_client_dict_list = [dict(d) for d in updated_normal_client_dict_list]

            iteration += 1

            if new_normal_count == normal_count:
                break

        # 恢复到最低方差比时刻的状态
        if best_state is not None:
            self._restore_client_states(best_state)
            self.normal_client_dict_list = best_normal_client_dict_list

        return {
            "min_variance_ratio": min_variance_ratio,
            "final_normal_client_dict_list": self.normal_client_dict_list,
            "total_iterations": iteration,
            "iteration_records": iteration_records
        }

    def train(self, epochs: int, learning_rate: float, aggregation: callable, save_path: str = None) -> None:
        """
        训练流程：
        1. 调用 iterative_update_normal_clients 确定最终的 normal_client_dict_list
        2. 进入 epochs 次循环，每次循环：
           - 将当前的 model 发送到每一个正常的 client 上
           - 通过 client.train() 收集各个正常 client 的梯度
           - 如果有 gradient_attack_func，则对梯度应用攻击
           - 将这些梯度打包为列表作为 aggregation 的传入参数
           - 利用返回值（整合后的梯度）对当前模型的参数进行更新
        """

        # 1. 确定最终的 normal_client_dict_list（筛选阶段）
        self.iterative_update_normal_clients()

        # 打印被剔除的异常 Client ID
        abnormal_ids = sorted([client.id for client in self.clients if not client.is_normal])
        normal_count = len(self.clients) - len(abnormal_ids)
        print(f"\n[Screening Result] Identified {len(abnormal_ids)} abnormal clients.")
        if len(abnormal_ids) > 0:
            print(f"   -> Abnormal Client IDs: {abnormal_ids}")
        print(f"[Screening Result] {normal_count} normal clients will participate in training.\n")

        # 2. 进入循环梯度更新
        normal_client_ids = [d['id'] for d in self.normal_client_dict_list]

        for epoch in range(epochs):
            # 2.1 将当前模型发送到正常客户端
            for client in self.clients:
                if client.id in normal_client_ids:
                    client.model = copy.deepcopy(self.model)

            # 2.2 收集正常客户端的梯度
            gradients_list = []
            for client in self.clients:
                if client.id in normal_client_ids:
                    gradient_result = client.train(local_epochs=1, learning_rate=learning_rate)
                    gradient_result['is_attacker'] = client.is_attacker
                    gradients_list.append(gradient_result)

            local_train_loss = sum([g['loss'] for g in gradients_list]) / len(gradients_list)

            # 2.3 梯度聚合（攻击已内嵌到 Client 内部，不再需要外层攻击函数）
            # 2.4 聚合梯度
            aggregated_gradients = aggregation(gradients_list)

            # 2.5 更新全局模型
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in aggregated_gradients:
                        param.data -= 1.0 * aggregated_gradients[name]

            # 2.6 计算聚合后的 loss
            self.model.eval()
            total_loss_after = 0.0
            total_samples = 0
            with torch.no_grad():
                for client in self.clients:
                    if client.id in normal_client_ids:
                        for batch_data in client.data:
                            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                                inputs, labels = batch_data
                            else:
                                continue
                            device = next(self.model.parameters()).device
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = self.model(inputs)
                            loss = F.cross_entropy(outputs, labels)
                            total_loss_after += loss.item() * inputs.size(0)
                            total_samples += inputs.size(0)
            avg_loss_after = total_loss_after / total_samples if total_samples > 0 else 0
            self.model.train()

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Local Train Loss (Avg): {local_train_loss:.6f} | "
                  f"Global Aggregated Model Loss: {avg_loss_after:.6f}")

        # 3. 保存模型参数
        if save_path is None:
            if hasattr(self.model, 'save_path') and self.model.save_path:
                save_path = self.model.save_path
            else:
                save_path = "model_params.pth"

        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")
