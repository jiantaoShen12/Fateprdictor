import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import collections
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from AE.utility import create_activation,compute_distance_matrix
sys.path.append('../')
'''
这段代码定义了一个多层感知机（MLP）、一个变分自编码器（VAE）模型的框架，以及一个自编码器（AutoEncoder）模型。下面是对代码的简要解释：

1. **MLP 类**：这是一个通用的多层感知机类，它接受一个层的列表（`layers_list`），一个dropout比例（`dropout`）
，是否使用批量归一化（`norm`），激活函数（`activation`），以及是否在最后一层应用激活函数（`last_act`）。
这个类构建了一个顺序模型，其中包含了线性层、可选的批量归一化层、激活层和dropout层。
2. **LatentModel 类**：这是变分自编码器中用于编码部分的类。它接受隐藏层的维度（`n_hidden`）
，潜在空间的维度（`n_latent`），KL散度的权重（`kl_weight`），以及预热步数（`warmup_step`）。
这个类定义了两个线性层，一个用于计算潜在空间的均值（`mu`），另一个用于计算对数方差（`logvar`）。它还实现了一个方法来调整KL散度权重。

3. **AutoEncoder 类**：这是一个自编码器模型类，它接受输入维度（`in_dim`），隐藏层的数量（`n_layers`）
，隐藏层的维度（`n_hidden`），潜在空间的维度（`n_latent`），激活函数的类型（`activate_type`），
dropout比例（`dropout`），是否使用批量归一化（`norm`），以及随机种子（`seed`）。这个类定义了编码器和解码器，
它们都是多层感知机，并且具有相同的架构。它还定义了获取潜在空间表示和从潜在空间生成数据的方法。

代码中还包含了一些未被使用的导入语句，如`from torchvision import datasets, transforms`，
这可能是因为原始代码中包含了图像处理的部分，但在这段代码中没有使用。
此外，代码中有一些注释掉的部分，如`# from AE.utility import create_activation,compute_distance_matrix`，
这表明原始代码可能依赖于一个名为`AE.utility`的自定义模块，但在这段代码中没有使用。

整体上，这段代码是一个用于构建和训练自编码器模型的框架，它可以用于数据的去噪、特征学习等任务。
'''
class MLP(nn.Module):
    def __init__(self, layers_list, dropout, norm,activation,last_act=False):
        super(MLP, self).__init__()
        layers=nn.ModuleList()
        assert len(layers_list)>=2, 'no enough layers'
        for i in range(len(layers_list)-2):
            layers.append(nn.Linear(layers_list[i],layers_list[i+1]))
            if norm:
                layers.append(nn.BatchNorm1d(layers_list[i+1]))
            if activation is not None:
                layers.append(activation)
            if dropout>0.:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layers_list[-2],layers_list[-1]))
        if norm:
            layers.append(nn.BatchNorm1d(layers_list[-1]))
        if last_act:
            if activation is not None:
                layers.append(activation)
        if dropout>0.:
            layers.append(nn.Dropout(dropout))
        # layers.append(nn.Linear(layers_list[-1],out_dim))
        self.network = nn.Sequential(*layers)
        # self.apply(init_weights_xavier_uniform)
    def forward(self,x):
        for layer in self.network:
            x=layer(x)
        return x
class LatentModel(nn.Module):
    def __init__(self,n_hidden,n_latent,
                 kl_weight=1e-6, warmup_step=10000):
        super(LatentModel,self).__init__()
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        # self.kl = 0
        self.kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step

    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = 0.0
        else:
            self.kl_weight = self.kl_weight + (1e-2 - 1e-6) / self.warmup_step

        # elif self.step_count == self.warmup_step:
        #     pass
            # self.step_count = 0
            # self.kl_weight = 1e-6

    def forward(self, h):
        mu = self.mu(h)
        log_var = self.logvar(h)
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        if self.training:
            z = mu + sigma * epsilon
            # (1 + log_var - mu ** 2 - log_var.exp()).sum()* self.kl_weight
            # print('hhhhhhh')
            self.kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum()  * self.kl_weight#/ z.shape[0]
            self.kl_schedule_step()
        else:
            z = mu
        return z





class AutoEncoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            n_layers: int = 1,
            n_hidden: int = 500,
            n_latent: int = 10,
            activate_type: str='relu',
            dropout: float = 0.2,
            norm: bool = False,
            seed: int=42,
    ):
        '''
        Autoencoder model.
        Encoder and Decoder take identical architectures.

        Parameters:
            in_dim:
                dimension of the input feature
            n_layers:
                number of hidden layers
            n_hidden:
                dimension of hidden layer. All hidden layers take the same dimensions
            n_latent:
                dimension of latent space
            activate_type:
                activation functions.
                Options: 'leakyrelu','relu', 'gelu', 'prelu', 'elu', and None for identity map.
            dropout:
                dropout rate
            norm:
                whether to include batch normalization layer
            seed:
                random seed.
        '''
        super(AutoEncoder,self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.in_dim=in_dim
        self.n_layers=n_layers
        self.n_hidden=n_hidden
        self.n_latent=n_latent
        self.dropout=dropout
        self.norm=norm
        self.activation=create_activation(activate_type)
        ## Encoder:
        self.encoder_layer=[in_dim]
        for i in range(n_layers):
            self.encoder_layer.append(n_hidden)
        self.encoder=MLP(self.encoder_layer,dropout,norm,self.activation,last_act=True)
        self.encoder_to_latent=MLP([self.encoder_layer[-1],n_latent],
                                   dropout,norm,self.activation)

        ## Decoder:
        self.decoder_layer=[n_latent]
        for i in range(n_layers):
            self.decoder_layer.append(n_hidden)
        self.decoder=MLP(self.decoder_layer,dropout,norm,self.activation,last_act=True)
        self.decoder_to_output=MLP([self.decoder_layer[-1],self.in_dim],dropout,norm,activation=None)


    def forward(self,x):
        rep=self.get_latent_representation(x,tensor=True)
        h = self.decoder(rep)
        x_recon=self.decoder_to_output(h)
        mse = nn.MSELoss(reduction='sum')
        loss = mse(x_recon, x)/x.shape[1]
        return loss


    def get_latent_representation(self,x,tensor:bool=False):
        '''
        Get latent space representation

        Parameters
        x:
            Input space
        tensor:
            If input x is a tensor, or it is a numpy array
        Return
        rep:
            latent space representation
            If tensor==True:
                return a tensor
            If tensor==Flase:
                return a numpy array
        '''
#        if not tensor:
#            x=torch.tensor(x,dtype=torch.float32)
#            self.eval()
        x=self.encoder(x)
        rep=self.encoder_to_latent(x)
        #if not tensor:
        #    rep=rep.detach().numpy()
        return rep
    def get_reconstruction(self, x):
        '''
        Reconstruct/impute gene expression data
        x:
            features. Numpy array
        Return
        x_recon:
            Numpy array
        '''
        self.eval()
        x=torch.tensor(x,dtype=torch.float32)
#        with torch.no_grad():
        x=self.encoder(x)
        x=self.encoder_to_latent(x)
        x = self.decoder(x)
        x_recon = self.decoder_to_output(x)

        #x_recon=x_recon.detach().numpy()
        return x_recon
    def get_generative(self,z):
        '''
        genereate gene expression data from latent space variable
        z:
            latent space representation. Numpy array
        Return
        x_recon:
            Numpy array
        '''
        self.eval()
        #z=torch.tensor(z,dtype=torch.float32)
#        with torch.no_grad():
        x = self.decoder(z)
        x_recon = self.decoder_to_output(x)
        #x_recon=x_recon.detach().numpy()
        return x_recon




