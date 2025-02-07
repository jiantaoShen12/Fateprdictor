import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import collections
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from AE.utility import create_activation
'''
这段代码定义了一个名为 `Trainer` 的类，用于训练和评估一个预训练模型（可能是自编码器，根据上下文）。`Trainer` 类的主要作用是封装模型训练和测试的流程，包括数据加载、训练循环、早停法（early stopping）等。下面是对代码的详细分析：

### 导入和辅助函数
- 导入了必要的 PyTorch 模块，包括数据加载、神经网络构建、优化器等。
- 导入了 `train_test_split` 函数，用于数据集的划分。
- 导入了 `create_activation` 函数，用于创建激活函数。
- 添加了当前目录的上级目录到系统路径，这通常是为了导入上层目录中的模块。

### `dataloader_split` 函数
这个函数接收原始数据集 `X`，以及其他参数如测试集大小、随机种子和批处理大小。它使用 `train_test_split` 来划分训练集和测试集，并创建 `TensorDataset` 和 `DataLoader` 对象，以便在训练和测试时使用。

### `Trainer` 类
`Trainer` 类包含以下主要部分：

#### 初始化 (`__init__`)
- 接收模型、数据集、测试集大小、批处理大小、学习率、权重衰减和随机种子等参数。
- 设置设备（CPU或CUDA）。
- 将模型移动到设备上。
- 使用 `dataloader_split` 函数创建训练和测试的数据加载器。
- 设置随机种子以确保结果的可复现性。
- 初始化 Adam 优化器。

#### 训练步骤 (`train_step`)
- 将模型设置为训练模式。
- 初始化训练损失为0。
- 遍历训练数据加载器中的所有批次，执行以下操作：
  - 将数据移动到设备上。
  - 清空梯度。
  - 计算损失并反向传播。
  - 更新损失值。
- 计算平均训练损失并返回。

#### 测试 (`test`)
- 将模型设置为评估模式。
- 初始化测试损失为0。
- 使用 `torch.no_grad()` 上下文管理器来禁用梯度计算，遍历测试数据加载器中的所有批次，计算损失并更新测试损失。
- 计算平均测试损失并返回。

#### 训练 (`train`)
- 初始化模型的历史记录字典，用于存储训练和验证损失以及epoch。
- 初始化最佳验证错误为无穷大，以及没有改进的epoch计数。
- 进行最大 `max_epoch` 次的训练，每次训练包括：
  - 调用 `train_step` 计算训练损失。
  - 调用 `test` 计算验证损失。
  - 更新模型历史记录。
  - 打印当前epoch的训练和验证损失。
  - 如果验证损失比之前的最佳验证损失有所下降，则更新最佳验证损失，并重置没有改进的epoch计数。如果没有改进，则增加计数。
  - 如果连续 `patient` 个epoch验证损失没有改进，则提前停止训练。
- 打印最佳验证误差。

### 输出
代码本身并没有直接的输出，但是会在训练过程中打印每个epoch的训练损失和验证损失。最终，训练完成后，会打印出最佳的验证误差。

### 总结
这段代码是一个通用的训练器，可以用于训练和评估任何PyTorch模型。它通过封装训练和测试过程，使得模型训练变得更加简单和模块化。通过早停法，可以避免过拟合，并在验证损失不再显著下降时提前结束训练，从而节省计算资源。
'''
sys.path.append('../')
def dataloader_split(X,test_size,seed,batch_size):
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(torch.tensor(X_train,dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
class Trainer(object):
    def __init__(self,
                 model,
                 X,
                 test_size:float=0.1,
                 batch_size:int=32,
                 lr:float=1e-3,
                 weight_decay:float=0.0,
                 seed:int=42,):
        '''
        Trainer for pretrain model.
        Parameters:
        model:
            A pytorch model defined in "models.py"
        X:
            Feature matrix. mxn numpy array.
                a standarized logorithmic data (i.e., zero mean, unit variance)
        test_size:
            fraction of testing/validation data size. default: 0.2
        batch_size:
            batch size.
        lr:
            learning rate.
        weight_decay:
            L2 regularization.
        seed:
            random seed.
        '''
        # self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        # if self.model.decoder_type=='normal':
        self.train_loader,self.test_loader=\
            dataloader_split(X,test_size,seed,batch_size)
        self.seed=seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
    def train_step(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(self.train_loader):
            data = data.to(self.device)
            # scale_factor = scale_factor.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(data,)
            loss.backward()

            train_loss += loss.item()#*data.shape[0]
            self.optimizer.step()
        train_loss=train_loss / len(self.train_loader.dataset)

        return train_loss
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(self.test_loader):
                data = data.to(self.device)
                loss =self.model(data,)
                test_loss += loss.item()#*data.shape[0]
        test_loss /= len(self.test_loader.dataset)
        return test_loss
    def train(self, max_epoch=500,tol=1e-2,  patient=30):
        # self.model.train()
        self.model.history = {'train_loss': [], 'val_loss': [],
                              'train_loss_ae':[],'val_loss_ae':[],
                              'train_loss_topo':[],'val_loss_topo':[],
                              'epoch':[]}
        best_val_error = float('inf')
        num_patient_epochs = 0
        for epoch in range(max_epoch):
            self.epoch=epoch
            # Train for one epoch and get the training loss
            train_loss = self.train_step()
            # Compute the validation error
            val_loss = self.test()
            self.model.history['train_loss'].append(train_loss)
            self.model.history['val_loss'].append(val_loss)
            self.model.history['epoch'].append(epoch)
            # if epoch % 10==0:
            print(f"Epoch {epoch}: train loss = {train_loss:.4f}, val error = {val_loss:.4f}")
            # Check if the validation error has decreased by at least tol
            if best_val_error - val_loss >= tol:
                best_val_error = val_loss
                num_patient_epochs = 0
            else:
                num_patient_epochs += 1
                # Check if we have exceeded the patience threshold
            if num_patient_epochs >= patient:
                print(f"No improvement in validation error for {patient} epochs. Stopping early.")
                break
        print(f"Best validation error = {best_val_error:.4f}")
