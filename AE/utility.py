import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init
from sklearn.metrics import pairwise_distances
'''
这段代码主要包含几个函数，用于在深度学习模型中设置激活函数、计算距离矩阵、计算最大成对距离以及初始化权重。下面是对每个函数的分析：

### `create_activation(name)`
这个函数根据提供的激活函数名称 `name` 返回相应的 PyTorch 激活层。支持的激活函数包括：
- `"relu"`：ReLU（线性修正单元）
- `"gelu"`：GELU（高斯误差线性单元）
- `"prelu"`：PReLU（参数化线性单元）
- `"elu"`：ELU（指数线性单元）
- `"leakyrelu"`：LeakyReLU（带有负斜率的ReLU）
- `"tanh"`：双曲正切函数

如果 `name` 为 `None`，则返回恒等激活（`nn.Identity()`），即不做任何非线性变换。

如果提供的激活函数名称不在上述列表中，则抛出一个 `NotImplementedError` 异常。

### `compute_distance_matrix(x, p=2)`
这个函数计算输入数据 `x` 的距离矩阵。`x` 可以是 PyTorch 张量或 NumPy 数组。距离矩阵是通过计算 `x` 中每一行之间的 p-范数距离来构建的。对于 PyTorch 张量，使用 `torch.norm` 计算距离；对于 NumPy 数组，使用 `pairwise_distances` 函数。

### `max_pairwise_distance(x, batch_size=1000)`
这个函数计算输入数据 `x`（可以是 NumPy 数组或 PyTorch 张量）的最大成对距离。它通过分批次（每批 `batch_size` 个样本）计算距离矩阵，并更新最大距离。注意，这个函数似乎在处理 NumPy 数组时使用 `pairwise_distances` 函数，但在注释中提到了 `np.fill_diagonal(distances, -np.inf)`，这行代码在实际代码中被注释掉了，因此它不会排除自距离。

### `init_weights_xavier_uniform(module)`
这个函数用于初始化 PyTorch 模型中的权重。如果 `module` 是 `nn.Linear`、`nn.Conv1d`、`nn.Conv2d` 或 `nn.Conv3d` 类型，它将使用 Xavier 均匀初始化方法初始化权重（`init.xavier_uniform_`），并将偏置初始化为零（`init.zeros_`）。

### 输出
这段代码本身不产生任何输出，因为它只定义了函数。但是，当这些函数在其他地方被调用时，它们将执行相应的操作，例如初始化模型层、计算距离矩阵等。实际的输出将取决于这些函数如何被使用。例如，如果 `create_activation` 被调用并传入一个有效的激活函数名称，它将返回一个 PyTorch 激活层对象；如果传入一个无效的名称，它将引发一个异常。其他函数在被调用时将执行计算并可能返回数值结果或初始化模型权重。
'''
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
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def compute_distance_matrix(x,p=2):
    if isinstance(x,torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    elif isinstance(x,np.ndarray):
        distances = pairwise_distances(x)
    else:
        raise NotImplementedError
    return distances


def max_pairwise_distance(x,batch_size=1000):
    n = len(x)
    max_distance = -np.inf
    for i in range(0, n, batch_size):
        if i+batch_size<n:
            batch = x[i:i + batch_size]
        else:
            batch = x[i:]
        # distances = np.linalg.norm(batch[:, None] - x, axis=2)
        distances = pairwise_distances(batch,x[i:])
        # np.fill_diagonal(distances, -np.inf)  # Set diagonal to -inf to exclude self-distances
        max_distance = max(max_distance, np.max(distances))
    return max_distance


def init_weights_xavier_uniform(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)



