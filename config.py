import torch

# ==============================
# 基础训练超参数 (Training Hyperparameters)
# ==============================
EPOCHS = 10                   # 训练轮数
LEARNING_RATE = 0.1          # 学习率
BATCH_SIZE = 64               # 批次大小 (根据MNIST常用设置调整)

# ==============================
# 联邦学习设置 (Federated Learning Settings)
# ==============================
NUM_CLIENTS = 20              # 客户端总数 (K=20)
AGGREGATION_METHOD = 'median' # 聚合方法: 'median' 或 'trimmed_mean'
TRIMMED_MEAN_BETA = 0.1       # trimmed_mean 的 beta 参数

# ==============================
# 攻击设置 (Attack Settings)
# ==============================
ATTACK_RATIO = 0.2            # 攻击者比例 (20% -> 4个客户端)

# ==============================
# 模型与路径设置 (Model & Paths)
# ==============================
# 模型保存路径
MODEL_SAVE_PATH = './cnn_model_params.pth'
# 基准模型保存路径（只训练一次，后续直接加载）
PROXY_MODEL_SAVE_PATH = './proxy_model.pth'
# 数据集下载路径
DATA_ROOT = './data'

# ==============================
# 运行设备 (Device)
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# 【新想法】服务器端信任根参数 (Server-side Clean Proxy)
# ==============================
# 用于训练基准模型的少量数据（信任根）
PROXY_DATA_RATIO = 0.05       # 用于训练基准模型的数据比例（相对于每个客户端的数据量）
PROXY_EPOCHS = 5              # 训练基准模型的轮数
PROXY_LR = 0.1                # 基准模型学习率

# ==============================
# 初始估计参数 (Init Theta - 新想法直接来自广播模型)
# ==============================
# 新想法：init_theta 直接来自广播的基准模型，不需要本地训练
INIT_EPOCHS = 0              # 新想法中不再需要本地训练来计算 init_theta
INIT_LR = 0.2                # 保留但不再使用

# ==============================
# 其他 (Misc)
# ==============================
SEED = 42                     # 随机种子
