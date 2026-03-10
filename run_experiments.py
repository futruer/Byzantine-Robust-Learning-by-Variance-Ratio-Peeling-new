"""
实验脚本：运行所有对比实验并保存结果
【新想法版本】使用服务器端代理数据进行恶意节点筛选
"""
import argparse
import torch
import random
import copy
import numpy as np
import pandas as pd
import os
import time
from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# 导入自定义模块
import config
import aggregation
from attack import cnn_rev_attack, gaussian_attack, sign_flipping_attack, scaling_attack
from cnn_model import NIST_CNN
from cnn_data_process import get_mnist_client_dataloaders, get_proxy_dataloader, cnn_reform_func_with_gh
from model import Client, Orchestrator


def print_header():
    """打印实验头部信息"""
    print("=" * 70)
    print("    Byzantine-Robust FL Experiments (新想法: 代理数据筛选)")
    print("=" * 70)
    print(f"  Dataset: MNIST")
    print(f"  Clients: {config.NUM_CLIENTS}")
    print(f"  Epochs per round: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Proxy Epochs: {config.PROXY_EPOCHS}")
    print(f"  Proxy LR: {config.PROXY_LR}")
    print("=" * 70)


def print_experiment_info(agg_name, attack_type, attack_ratio, exp_num, total):
    """打印当前实验的详细信息"""
    agg_descriptions = {
        'peeling': 'VR Peeling (新想法：代理数据+VR Peeling)',
        'fedavg': 'FedAvg (无防御基准：不筛选，直接均值聚合)',
        'trimmed_mean': f'Trimmed Mean (β={config.TRIMMED_MEAN_BETA})',
        'multi_krum': 'Multi-Krum (n_remove=2, n_select=5)',
        'clustering': 'Clustering-based (基于聚类)',
        'median': 'Coordinate-wise Median (坐标级中位数)'
    }

    attack_descriptions = {
        'label_flipping': 'Label Flipping (标签翻转攻击)',
        'gaussian': 'Gaussian Noise (高斯噪声攻击)',
        'sign_flipping': 'Sign Flipping (符号翻转攻击)',
        'scaling': 'Scaling (缩放攻击)'
    }

    print("\n" + "=" * 70)
    print(f"  Experiment [{exp_num}/{total}]")
    print("-" * 70)
    print(f"  [聚合方法] {agg_name:15s} - {agg_descriptions.get(agg_name, '')}")
    print(f"  [攻击类型] {attack_type:15s} - {attack_descriptions.get(attack_type, '')}")
    print(f"  [攻击比例] {attack_ratio*100:.0f}%  ({int(config.NUM_CLIENTS * attack_ratio)}/{config.NUM_CLIENTS} clients)")
    print("=" * 70)


def set_seed(seed):
    """固定随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_attack_func(attack_type):
    """获取攻击函数"""
    attacks = {
        'label_flipping': None,
        'gaussian': gaussian_attack,
        'sign_flipping': sign_flipping_attack,
        'scaling': scaling_attack,
        'mixed': sign_flipping_attack
    }
    return attacks.get(attack_type, None)


def get_aggregation_func(agg_name):
    """获取聚合函数"""
    max_attackers = int(config.NUM_CLIENTS * 0.4)

    aggs = {
        'peeling': partial(aggregation.coordinate_wise_median),
        'fedavg': aggregation.fedavg,
        'trimmed_mean': partial(aggregation.coordinate_wise_trimmed_mean, beta=config.TRIMMED_MEAN_BETA),
        'multi_krum': partial(aggregation.multi_krum, n_remove=max_attackers, n_select=5),
        'clustering': aggregation.clustering_based,
        'median': aggregation.coordinate_wise_median,
    }
    return aggs.get(agg_name, None)


def evaluate_model(model, device, data_root='./data'):
    """在测试集上评估模型准确率"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def run_benchmark_training(model, clients, epochs, learning_rate, aggregation_func, gradient_attack_func, device, verbose=True):
    """
    Benchmark 方法的训练循环：【绝不筛选】，直接接收所有客户端梯度进行聚合
    """
    if verbose:
        print(f"       开始 {epochs} 轮 Benchmark 训练 (无筛选)...")

    for epoch in range(epochs):
        for client in clients:
            client.model = copy.deepcopy(model)

        gradients_list = []
        for client in clients:
            gradient_result = client.train(local_epochs=1, learning_rate=learning_rate)
            gradient_result['is_attacker'] = client.is_attacker
            gradients_list.append(gradient_result)

        local_train_loss = sum([g['loss'] for g in gradients_list]) / len(gradients_list)

        # 攻击已内嵌到 Client 内部，不再需要外层攻击函数

        aggregated_gradients = aggregation_func(gradients_list)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= 1.0 * aggregated_gradients[name]

        model.eval()
        total_loss_after = 0.0
        total_samples = 0
        with torch.no_grad():
            for client in clients:
                if not client.is_attacker:
                    for batch_data in client.data:
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            inputs, labels = batch_data
                        else:
                            continue
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = F.cross_entropy(outputs, labels)
                        total_loss_after += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)

        avg_loss_after = total_loss_after / total_samples if total_samples > 0 else 0
        model.train()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Local Train Loss (Avg): {local_train_loss:.6f} | "
                  f"Global Aggregated Model Loss: {avg_loss_after:.6f}")

    if verbose:
        print(f"       Benchmark 训练完成")


def run_experiment(agg_name, attack_type, attack_ratio, epochs, device='cpu', seed=42, verbose=True):
    set_seed(seed)
    start_time = time.time()

    if verbose:
        print("\n  [1/6] 准备数据...")

    raw_client_data = get_mnist_client_dataloaders(
        num_clients=config.NUM_CLIENTS,
        batch_size=config.BATCH_SIZE,
        root=config.DATA_ROOT
    )

    # 【新想法】获取服务器端代理数据
    proxy_data_loader = get_proxy_dataloader(
        batch_size=config.BATCH_SIZE,
        root=config.DATA_ROOT
    )

    if verbose:
        print(f"       数据准备完成，共 {len(raw_client_data)} 个客户端")
        print(f"       代理数据准备完成")

    gradient_attack_func = None

    if verbose:
        print(f"  [2/6] 设置攻击: {attack_type} (比例: {attack_ratio*100:.0f}%)")

    if attack_type == 'label_flipping':
        attacked_client_data = cnn_rev_attack(raw_client_data, alpha=attack_ratio)
    elif attack_type == 'mixed':
        from attack import LabelFlippedDataset
        random.seed(42)
        K = len(raw_client_data)

        num_lf = int(K * attack_ratio / 2)
        num_grad = int(K * attack_ratio) - num_lf

        all_attackers = random.sample(range(K), num_lf + num_grad)
        lf_indices = set(all_attackers[:num_lf])
        grad_indices = set(all_attackers[num_lf:])

        attacked_client_data = []
        for i, client_dict in enumerate(raw_client_data):
            new_dict = client_dict.copy()
            if i in lf_indices:
                orig_loader = client_dict['dataloader']
                new_loader = DataLoader(LabelFlippedDataset(orig_loader.dataset),
                                        batch_size=orig_loader.batch_size, shuffle=True)
                new_dict['dataloader'] = new_loader
                new_dict['is_attacker'] = False
                new_dict['is_lf_attacker'] = True
            elif i in grad_indices:
                new_dict['is_attacker'] = True
                new_dict['is_lf_attacker'] = False
            else:
                new_dict['is_attacker'] = False
                new_dict['is_lf_attacker'] = False

            attacked_client_data.append(new_dict)

        gradient_attack_func = get_attack_func('mixed')
    else:
        random.seed(42)
        K = len(raw_client_data)
        num_attackers = int(K * attack_ratio)
        attacker_indices = set(random.sample(range(K), num_attackers))

        attacked_client_data = []
        for i, client_dict in enumerate(raw_client_data):
            new_dict = client_dict.copy()
            new_dict['is_attacker'] = (i in attacker_indices)
            attacked_client_data.append(new_dict)

        gradient_attack_func = get_attack_func(attack_type)
        if gradient_attack_func is None:
            raise ValueError(f"Unknown attack type: {attack_type}")

    if verbose:
        if attack_type == 'mixed':
            lf_ids = sorted([c['id'] for c in attacked_client_data if c.get('is_lf_attacker', False)])
            grad_ids = sorted([c['id'] for c in attacked_client_data if c.get('is_attacker', False)])
            print(f"       混合攻击者总数: {len(lf_ids) + len(grad_ids)}/{len(attacked_client_data)}")
            print(f"       -> 数据投毒 (Label Flip) IDs: {lf_ids}")
            print(f"       -> 梯度投毒 (Sign Flip) IDs : {grad_ids}")
        else:
            attacker_count = sum(1 for c in attacked_client_data if c.get('is_attacker', False))
            attacker_ids = sorted([c['id'] for c in attacked_client_data if c.get('is_attacker', False)])
            print(f"       攻击者数量: {attacker_count}/{len(attacked_client_data)}")
            print(f"       攻击者ID: {attacker_ids}")

    if verbose:
        print(f"  [3/6] 初始化模型和客户端...")

    global_model = NIST_CNN().to(device)

    clients = []
    for client_info in attacked_client_data:
        c_id = client_info['id']
        c_loader = client_info['dataloader']

        client = Client(
            model=global_model,
            data=c_loader,
            reform_func_with_gh=cnn_reform_func_with_gh,
            id=c_id,
            is_attacker=client_info.get('is_attacker', False),
            attack_type=attack_type
        )
        clients.append(client)

    agg_func = get_aggregation_func(agg_name)
    if agg_func is None:
        raise ValueError(f"Unknown aggregation: {agg_name}")

    use_screening = (agg_name == 'peeling')

    if verbose:
        screening_info = "启用筛选（新想法：代理数据+VR Peeling）" if use_screening else "不筛选（Benchmark 全部收录）"
        print(f"  [4/6] 训练模式: {screening_info}")

    if use_screening:
        # 【新想法】：使用代理数据训练的基准模型进行筛选
        orchestrator = Orchestrator(
            model=global_model,
            clients=clients,
            proxy_data_loader=proxy_data_loader,
            gradient_attack_func=gradient_attack_func
        )
        orchestrator.train(
            epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            aggregation=agg_func,
            save_path=None
        )
    else:
        # Benchmark 方法
        run_benchmark_training(
            model=global_model,
            clients=clients,
            epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            aggregation_func=agg_func,
            gradient_attack_func=gradient_attack_func,
            device=device,
            verbose=verbose
        )

    # 5. 评估
    if verbose:
        print(f"  [5/6] 评估模型...")

    accuracy = evaluate_model(global_model, device, config.DATA_ROOT)
    elapsed_time = time.time() - start_time

    if verbose:
        print(f"  [6/6] 训练完成! 测试准确率: {accuracy:.2f}%")
        print(f"       耗时: {elapsed_time:.2f}秒")

    return accuracy, elapsed_time


def run_experiments(output_file='results_new_frame.csv'):
    device = config.DEVICE
    print_header()

    agg_methods = ['fedavg', 'trimmed_mean', 'multi_krum', 'clustering', 'peeling', 'median']
    attack_types = ['label_flipping', 'sign_flipping', 'scaling', 'gaussian']
    attack_ratios = [0.3, 0.4]
    epochs = config.EPOCHS

    results = []
    valid_experiments = []
    for agg in agg_methods:
        for attack in attack_types:
            for ratio in attack_ratios:
                if ratio == 0.0 and attack != 'label_flipping':
                    continue
                valid_experiments.append((agg, attack, ratio))

    total_experiments = len(valid_experiments)
    current = 0

    print(f"\n>>> 开始运行 {total_experiments} 个实验配置...")
    print(f">>> 聚合方法: {', '.join(agg_methods)}")
    print(f">>> 攻击类型: {', '.join(attack_types)}")
    print(f">>> 攻击比例: {[f'{r*100:.0f}%' for r in attack_ratios]}")
    print()

    total_start_time = time.time()

    for agg, attack, ratio in valid_experiments:
        current += 1
        print_experiment_info(agg, attack, ratio, current, total_experiments)

        try:
            accuracy, elapsed_time = run_experiment(
                agg_name=agg,
                attack_type=attack,
                attack_ratio=ratio,
                epochs=epochs,
                device=device,
                seed=config.SEED,
                verbose=True
            )

            print("\n" + "=" * 70)
            print(f"  ✓ 实验完成 | 准确率: {accuracy:.2f}% | 耗时: {elapsed_time:.2f}s")
            print("=" * 70)

            results.append({
                'aggregation': agg,
                'attack_type': attack,
                'attack_ratio': ratio,
                'accuracy': accuracy,
                'time': elapsed_time
            })
        except Exception as e:
            print(f"\n  ✗ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'aggregation': agg,
                'attack_type': attack,
                'attack_ratio': ratio,
                'accuracy': None,
                'error': str(e)
            })

    total_time = time.time() - total_start_time

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("                     实验完成 - 汇总信息")
    print("=" * 70)
    print(f"  总实验数: {total_experiments}")
    print(f"  总耗时: {total_time/60:.2f} 分钟")
    print(f"  平均每实验: {total_time/total_experiments:.2f} 秒")
    print(f"  结果保存: {output_file}")
    print("=" * 70)

    if len(results) > 0:
        print("\n  结果预览:")
        print("-" * 70)
        print(f"  {'聚合方法':<15} {'攻击类型':<15} {'攻击比例':<10} {'准确率':<10}")
        print("-" * 70)
        for r in results:
            acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else "N/A"
            print(f"  {r['aggregation']:<15} {r['attack_type']:<15} {r['attack_ratio']*100:.0f}%{'':<5} {acc_str:<10}")
        print("-" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(description="Byzantine-Robust FL Experiments (New Frame)")
    parser.add_argument('--output', type=str, default='results_new_frame.csv', help='输出 CSV 文件路径')
    parser.add_argument('--device', type=str, default=config.DEVICE, help='运行设备')
    args = parser.parse_args()

    run_experiments(output_file=args.output)


if __name__ == "__main__":
    main()
