import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from TorchDiffEqPack import odesolve
import sys
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from torchdiffeq import odeint
from functools import partial
import getpass
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
'''
这段代码定义了一个基于神经网络的模型，用于解决与时间序列数据相关的特定问题，例如单细胞RNA测序数据分析中的动态过程建模。代码中实现的主要内容包括：
1. **参数配置** (`create_args` 函数): 通过用户输入来设置模型训练和模拟的参数，如数据集名称、时间点、学习率、隐藏层维度等。
2. **神经网络定义** (`UOT`, `HyperNetwork1`, `HyperNetwork2` 类): 
定义了用于模拟时间序列动态的神经网络结构，包括用于计算时间导数 `v=dx/dt` 和生长项 `g` 的超网络。
3. **雅可比矩阵追踪** (`trace_df_dz` 函数): 计算雅可比矩阵的迹，用于分析动态系统的稳定性。
4. **权重初始化** (`initialize_weights` 函数): 初始化网络权重，通常使用Xavier初始化方法。
5. **运行平均值计算** (`RunningAverageMeter` 类): 用于计算训练过程中的损失和其他指标的运行平均值。
6. **多模态高斯密度函数** (`MultimodalGaussian_density` 函数): 用于计算多模态高斯分布的概率密度函数，可能用于模型的概率解释或评估。
7. **采样函数** (`Sampling` 函数): 根据多模态高斯分布进行采样，可能用于生成训练数据或进行概率推断。
8. **数据加载** (`loaddata` 函数): 加载数据集，将数据从 `.npy` 文件读取到内存中，并转换为适合模型输入的格式。
9. **生长率计算** (`ggrowth` 函数): 计算给定时间点的生长率，可能用于模拟细胞生长或其他生物过程。
10. **转换损失计算** (`trans_loss` 函数): 计算状态转换过程中的损失，可能用于优化模型以更好地模拟状态转换。
11. **最大公约数计算** (`gcd_list` 函数): 计算列表中数字的最大公约数，可能用于确定时间步长或其他周期性参数。
12. **模型训练** (`train_model` 函数): 实现模型的训练过程，包括损失计算、梯度更新、学习率调整等。
13. **三维可视化** (`plot_3d` 函数): 可视化推断出的细胞轨迹，使用matplotlib和plotly库生成三维图。
14. **雅可比矩阵和梯度可视化** (`plot_jac_v`, `plot_grad_g` 函数): 可视化雅可比矩阵和梯度，帮助分析模型的动态特性。
代码中还包含了一些辅助函数，如 `odeint` 和 `odesolve`，它们用于实现常微分方程的数值解。整体上，这段代码是一个用于模拟和分析时间序列数据（如单细胞发展过程）的复杂系统，可能在生物学研究中用于理解细胞状态的变化和转变。
'''
class Args:
    pass

def create_args():
    args = Args()
    args.dataset = input("Name of the data set. Options: EMT; Lineage; Bifurcation; Simulation (default: EMT): ") or 'EMT'
    n = int(input("Enter the number of time points (default: 5): ") or 5)
    args.timepoints =timess =[(i+1) / (n) for i in range(n)]
   #timepoints = input("Time points of data (default: 0, 0.1, 0.3, 0.9, 2.1): ")
   #args.timepoints = [float(tp.strip()) for tp in timepoints.split(",")] if timepoints else [0, 0.1, 0.3, 0.9, 2.1]
    args.niters = int(input("Number of training iterations (default: 5000): ") or 5000)
    args.lr = float(input("Learning rate (default: 3e-3): ") or 3e-3)
    args.num_samples = int(input("Number of sampling points per epoch (default: 100): ") or 100)
    args.hidden_dim = int(input("Dimension of the hidden layer (default: 16): ") or 16)
    args.n_hiddens = int(input("Number of hidden layers (default: 4): ") or 4)
    args.activation = input("Activation function (default: Tanh): ") or 'Tanh'
    args.gpu = int(input("GPU device index (default: 0): ") or 0)
    args.input_dir = input("Input Files Directory (default: Input/): ") or 'Input/'
    args.save_dir = input("Output Files Directory (default: Output/EMT/): ") or 'Output/EMT/'
    args.seed = int(input("Random seed (default: 1): ") or 1)
    return args


class UOT(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.hyper_net1 = HyperNetwork1(in_out_dim, hidden_dim, n_hiddens,activation) #v= dx/dt
        self.hyper_net2 = HyperNetwork2(in_out_dim, hidden_dim, activation) #g

    def forward(self, t, states):
        z = states[0]
        g_z = states[1]
        logp_z = states[2]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.hyper_net1(t, z)
            
            g = self.hyper_net2(t, z)

            dlogp_z_dt = g - trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, g, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x
    

class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def MultimodalGaussian_density(x,time_all,time_pt,data_train,sigma,device):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/num_gaussian
    return p_n


    
def Sampling(num_samples,time_all,time_pt,data_train,sigma,device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples


def loaddata(args,device):
    data=np.load(os.path.join(args.input_dir,(args.dataset+'.npy')),allow_pickle=True)
    data_train=[]
    for i in range(data.shape[1]):
        data_train.append(torch.from_numpy(data[0,i]).type(torch.float32).to(device))
    return data_train


def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)                       
    gg = func.forward(t, y)[1]
    return (y_0,y_00,gg)
    
    
def trans_loss(t,y,func,device,odeint_setp):
    outputs= func.forward(t, y)
    v = outputs[0]
    g = outputs[1]
    y_0 = torch.zeros(g.shape).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    g_growth = partial(ggrowth,func=func,device=device)
    if torch.is_nonzero(t):
        _,_, exp_g = odeint(g_growth, (y_00,y_0,y_0), torch.tensor([0,t]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
        f_int = (torch.norm(v,dim=1)**2+torch.norm(g,dim=1)**2).unsqueeze(1)*torch.exp(exp_g[-1])
        return (y_00,y_0,f_int)
    else:
        return (y_00,y_0,y_0)

def gcd_list(numbers):
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    gcd_value = numbers[0]
    for i in range(1, len(numbers)):
        gcd_value = _gcd(gcd_value, numbers[i])

    return gcd_value


def train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr):
    warnings.filterwarnings("ignore")

    loss = 0
    L2_value1 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value2 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    for i in range(len(train_time)-1): 
        x = Sampling(args.num_samples, train_time,i+1,data_train,0.02,device)
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
        g_t1 = logp_diff_t1
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[0]})
        z_t0, g_t0, logp_diff_t0 = odesolve(func,y0=(x, g_t1, logp_diff_t1),options=options)
        aa = MultimodalGaussian_density(z_t0, train_time, 0, data_train,sigma_now,device) #normalized density
        
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        
        aaa = MultimodalGaussian_density(x, train_time, i+1, data_train,sigma_now,device) * torch.tensor(data_train[i+1].shape[0]/data_train[0].shape[0]) # mass
        
        L2_value1[0][i] = mse(aaa,torch.exp(logp_x.view(-1)))
        
        loss = loss  + L2_value1[0][i]*1e4 
        
        # loss between each two time points
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[i]})
        z_t0, g_t0, logp_diff_t0= odesolve(func,y0=(x, g_t1, logp_diff_t1),options=options)
        
        aa = MultimodalGaussian_density(z_t0, train_time, i, data_train,sigma_now,device)* torch.tensor(data_train[i].shape[0]/data_train[0].shape[0])
        
        #find zero density
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        
        L2_value2[0][i] = mse(aaa,torch.exp(logp_x.view(-1))) 
        loss = loss  + L2_value2[0][i]*1e4 
        
        
    # compute transport cost efficiency
    transport_cost = partial(trans_loss,func=func,device=device,odeint_setp=odeint_setp)
    x0 = Sampling(args.num_samples,train_time,0,data_train,0.02,device) 
    logp_diff_t00 = torch.zeros(x0.shape[0], 1).type(torch.float32).to(device)
    g_t00 = logp_diff_t00
    _,_,loss1 = odeint(transport_cost,y0=(x0, g_t00, logp_diff_t00),t = torch.tensor([0, integral_time[-1]]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
    loss = loss + integral_time[-1]*loss1[-1].mean(0)


    if (itr >1):
        if ((itr % 100 == 0) and (itr<=args.niters-400) and (sigma_now>0.02) and (L2_value1.mean()<=0.0003)):
            sigma_now = sigma_now/2

    return loss, loss1, sigma_now, L2_value1, L2_value2


def plot_3d(func, data_train, train_time, integral_time, args, device):
    viz_samples = 20
    sigma_a = 0.001

    t_list = []  # list(reversed(integral_time))#integral_time #np.linspace(5, 0, viz_timesteps)
    # options.update({'t_eval':t_list})

    z_t_samples = []
    z_t_data = []
    v = []
    g = []
    t_list2 = []
    odeint_setp = gcd_list([num * 100 for num in integral_time]) / 100
    integral_time2 = np.arange(integral_time[0], integral_time[-1] + odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals=2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time), integral_time))[0]
    sample_time = list(reversed(sample_time))

    with torch.no_grad():
        for i in range(len(integral_time)):
            z_t0 = data_train[i]

            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])

        # traj backward
        z_t0 = Sampling(viz_samples, train_time, len(train_time) - 1, data_train, sigma_a, device)
        # z_t0 = z_t0[z_t0[:,2]>1]
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_t0, g0, logp_diff_t0))[
            0]  # True_v(z_t0)
        g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_t0, g0, logp_diff_t0))[1]

        v.append(v_t.cpu().detach().numpy())
        g.append(g_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})

        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval': plot_time})
        z_t1, _, logp_diff_t1 = odesolve(func, y0=(z_t0, g0, logp_diff_t0), options=options)
        for i in range(len(plot_time) - 1):
            v_t = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[
                0]  # True_v(z_t0)
            g_t = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[
                1]

            z_t_samples.append(z_t1[i + 1].cpu().detach().numpy())
            g.append(g_t.cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())
            t_list.append(plot_time[i + 1])

        aa = 5  # 3
        angle1 = 10  # 30
        angle2 = 75  # 30
        trans = 0.8
        trans2 = 0.4
        widths = 0.2  # arrow width
        ratio1 = 0.4
        fig = plt.figure(figsize=(4 * 2, 3 * 2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 8

        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        # fig.suptitle(f'{t:.1f}day')
        ax1 = plt.axes(projection='3d')
        #ax1.grid(False)
        ax1.set_xlabel('AE1')
        ax1.set_ylabel('AE2')
        ax1.set_zlabel('AE3')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_zlim(-2, 2)
        ax1.set_xticks([-2, 2])
        ax1.set_yticks([-2, 2])
        ax1.set_zticks([-2, 2])
        ax1.view_init(elev=angle1, azim=angle2)
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.invert_xaxis()
        ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))
        line_width = 0.3

        color_wanted = [np.array([250, 187, 110]) / 255,
                        np.array([173, 219, 136]) / 255,
                        np.array([250, 199, 179]) / 255,
                        np.array([238, 68, 49]) / 255,
                        np.array([206, 223, 239]) / 255,
                        np.array([3, 149, 198]) / 255,
                        np.array([180, 180, 213]) / 255,
                        np.array([178, 143, 237]) / 255]
        for j in range(viz_samples):  # individual traj
            for i in range(len(plot_time) - 1):
                ax1.plot([z_t_samples[i][j, 0], z_t_samples[i + 1][j, 0]],
                         [z_t_samples[i][j, 1], z_t_samples[i + 1][j, 1]],
                         [z_t_samples[i][j, 2], z_t_samples[i + 1][j, 2]],
                         linewidth=0.5, color='grey', zorder=2)

        # add inferrred trajecotry
        for i in range(len(sample_time)):
            ax1.scatter(z_t_samples[sample_time[i]][:, 0], z_t_samples[sample_time[i]][:, 1],
                        z_t_samples[sample_time[i]][:, 2], s=aa * 10, linewidth=0, color=color_wanted[i], zorder=3)
            ax1.quiver(z_t_samples[sample_time[i]][:, 0], z_t_samples[sample_time[i]][:, 1],
                       z_t_samples[sample_time[i]][:, 2],
                       v[sample_time[i]][:, 0] / v_scale, v[sample_time[i]][:, 1] / v_scale,
                       v[sample_time[i]][:, 2] / v_scale, color='k', alpha=1, linewidths=widths * 2,
                       arrow_length_ratio=0.3, zorder=4)

        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:, 0], z_t_data[i][:, 1], z_t_data[i][:, 2], s=aa, linewidth=line_width, alpha=0.7,
                        facecolors='none', edgecolors=color_wanted[i], zorder=1)

        # plt.savefig(os.path.join(args.save_dir, f"traj_3d.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        plt.show()

# # plot 3d of inferred trajectory of 20 cells
# def plot_3d(func,data_train,train_time,integral_time,args,device):
#     viz_samples = 25
#     sigma_a = 0.001
#
#     t_list = []#list(reversed(integral_time))#integral_time #np.linspace(5, 0, viz_timesteps)
#     #options.update({'t_eval':t_list})
#
#     z_t_samples = []
#     z_t_data = []
#     v = []
#     g = []
#     t_list2 = []
#     odeint_setp = gcd_list([num * 100 for num in integral_time])/100
#     integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
#     integral_time2 = np.round_(integral_time2, decimals = 2)
#     plot_time = list(reversed(integral_time2))
#     sample_time = np.where(np.isin(np.array(plot_time),integral_time))[0]
#     sample_time = list(reversed(sample_time))
#
#     with torch.no_grad():
#         for i in range(len(integral_time)):
#
#             z_t0 =  data_train[i]
#
#             z_t_data.append(z_t0.cpu().detach().numpy())
#             t_list2.append(integral_time[i])
#
#         # traj backward
#         z_t0 =  Sampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
#         #z_t0 = z_t0[z_t0[:,2]>1]
#         logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
#         g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
#         v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[0] #True_v(z_t0)
#         g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[1]
#
#         v.append(v_t.cpu().detach().numpy())
#         g.append(g_t.cpu().detach().numpy())
#         z_t_samples.append(z_t0.cpu().detach().numpy())
#         t_list.append(plot_time[0])
#         options = {}
#         options.update({'method': 'Dopri5'})
#         options.update({'h': None})
#         options.update({'rtol': 1e-3})
#         options.update({'atol': 1e-5})
#         options.update({'print_neval': False})
#         options.update({'neval_max': 1000000})
#         options.update({'safety': None})
#
#         options.update({'t0': integral_time[-1]})
#         options.update({'t1': 0})
#         options.update({'t_eval':plot_time})
#         z_t1,_, logp_diff_t1= odesolve(func,y0=(z_t0,g0, logp_diff_t0),options=options)
#         for i in range(len(plot_time)-1):
#             v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[0] #True_v(z_t0)
#             g_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[1]
#
#             z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
#             g.append(g_t.cpu().detach().numpy())
#             v.append(v_t.cpu().detach().numpy())
#             t_list.append(plot_time[i+1])
#
#         aa=5#3
#         angle1 = 10#30
#         angle2 = 75#30
#         trans = 0.8
#         trans2 = 0.4
#         widths = 0.2 #arrow width
#         ratio1 = 0.4
#         fig = plt.figure(figsize=(4*2,3*2), dpi=200)
#         plt.tight_layout()
#         plt.margins(0, 0)
#         v_scale = 10
#
#
#         plt.tight_layout()
#         plt.axis('off')
#         plt.margins(0, 0)
#         #fig.suptitle(f'{t:.1f}day')
#         ax1 = plt.axes(projection ='3d')
#         ax1.grid(False)
#         ax1.set_xlabel('AE1')
#         ax1.set_ylabel('AE2')
#         ax1.set_zlabel('AE3')
#         ax1.set_xlim(-2,2)
#         ax1.set_ylim(-2,2)
#         ax1.set_zlim(-2,2)
#         ax1.set_xticks([-2,2])
#         ax1.set_yticks([-2,2])
#         ax1.set_zticks([-2,2])
#         ax1.view_init(elev=angle1, azim=angle2)
#
#         ax1.xaxis.set_pane_color((1, 1, 1, 0))
#         ax1.yaxis.set_pane_color((1, 1, 1, 0))
#         ax1.zaxis.set_pane_color((1, 1, 1, 0))
#
#         ax1.invert_xaxis()
#         ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))
#         line_width = 0.3
#
#
#
#         color_wanted = [np.array([173,219,136])/255,
#                         np.array([200, 150, 100])/255,
#                         np.array([238, 68, 49]) / 255,
#                         np.array([180,180,213])/255,
#                         np.array([250,187,110])/255,
#                         np.array([3,149,198])/255,
#                         np.array([178,143,237])/255,
#                         np.array([250,199,179])/255,
# ]
#         for j in range(viz_samples): #individual traj
#             for i in range(len(plot_time)-1):
#                 ax1.plot([z_t_samples[i][j,0],z_t_samples[i+1][j,0]],
#                             [z_t_samples[i][j,1],z_t_samples[i+1][j,1]],
#                             [z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
#                             linewidth=0.5,color ='grey',zorder=2)
#
#
#         # add inferrred trajecotry
#         for i in range(len(sample_time)):
#             ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],s=aa*10,linewidth=0, color=color_wanted[i],zorder=3)
#             ax1.quiver(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],
#                        v[sample_time[i]][:,0]/v_scale,v[sample_time[i]][:,1]/v_scale,v[sample_time[i]][:,2]/v_scale, color='k',alpha=1,linewidths=widths*2,arrow_length_ratio=0.12,zorder=4)
#
#
#         for i in range(len(integral_time)):
#             ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],z_t_data[i][:,2],s=aa,linewidth=line_width,alpha = 0.7, facecolors='none', edgecolors=color_wanted[i],zorder=1)
#
#         plt.savefig(os.path.join(args.save_dir, f"traj_3d.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
#         plt.show()
#

def Jacobian(f, z):
    """Calculates Jacobian df/dz.
    """
    jac = []
    for i in range(z.shape[1]):
        df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]),retain_graph=True, create_graph=True)[0].view(z.shape[0], -1)
        jac.append(torch.unsqueeze(df_dz, 1))
    jac = torch.cat(jac, 1)
    return jac

# plot avergae jac of v of cells (z_t) at time (time_pt)
def plot_jac_v(func,z_t,time_pt,title,gene_list,args,device):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    # compute the mean of jacobian of v within cells z_t at time (time_pt)
    dim = z_t.shape[1]
    jac = np.zeros((dim,dim))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        v_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0))[0]
        jac = jac+Jacobian(v_xt, x_t).reshape(dim,dim).detach().cpu().numpy()
    jac = jac/z_t.shape[0]
    
    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.add_subplot(111)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0, 0)
    ax.set_title('Jacobian of velocity')
    sns.heatmap(jac,cmap="coolwarm",xticklabels=gene_list,yticklabels=gene_list)
    ax.set_xticks([])  # Remove x-axis tick marks
    ax.set_yticks([])  # Remove y-axis tick marks
    ax.axis('off')
    plt.savefig(os.path.join(args.save_dir, title),format="pdf",
                pad_inches=0.2, bbox_inches='tight')
    plt.show()
                

# plot avergae gradients of g of cells (z_t) at time (time_pt)
def plot_grad_g(func,z_t,time_pt,title,gene_list,args,device):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    dim = z_t.shape[1]
    gg = np.zeros((dim,dim))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        g_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0))[1]
        gg = gg+torch.autograd.grad(g_xt, x_t, torch.ones_like(g_xt),retain_graph=True, create_graph=True)[0].view(x_t.shape[0], -1).reshape(dim,1).detach().cpu().numpy()
    gg = gg/z_t.shape[0]
    
    fig= plt.figure(figsize=(1, 4), dpi=200)
    ax = fig.add_subplot(111)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0, 0)
    ax.set_title('Gradient of growth')
    sns.heatmap(gg,cmap="coolwarm",xticklabels=[],yticklabels=gene_list)
    ax.set_xticks([])  # Remove x-axis tick marks
    ax.set_yticks([])  # Remove y-axis tick marks
    ax.axis('off') 
    plt.savefig(os.path.join(args.save_dir, title),format="pdf",
                pad_inches=0.2, bbox_inches='tight')
    plt.show()


def plot_3d1(func, data_train, train_time, integral_time, args, device):
    viz_samples = 20
    sigma_a = 0.001

    t_list = []  # list(reversed(integral_time))#integral_time #np.linspace(5, 0, viz_timesteps)
    # options.update({'t_eval':t_list})

    z_t_samples = []
    z_t_data = []
    v = []
    g = []
    t_list2 = []
    odeint_setp = gcd_list([num * 100 for num in integral_time]) / 100
    integral_time2 = np.arange(integral_time[0], integral_time[-1] + odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals=2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time), integral_time))[0]
    sample_time = list(reversed(sample_time))

    with torch.no_grad():
        for i in range(len(integral_time)):
            z_t0 = data_train[i]

            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])

        # traj backward
        z_t0 = Sampling(viz_samples, train_time, len(train_time) - 1, data_train, sigma_a, device)
        # z_t0 = z_t0[z_t0[:,2]>1]
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_t0, g0, logp_diff_t0))[
            0]  # True_v(z_t0)
        g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_t0, g0, logp_diff_t0))[1]

        v.append(v_t.cpu().detach().numpy())
        g.append(g_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})

        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval': plot_time})
        z_t1, _, logp_diff_t1 = odesolve(func, y0=(z_t0, g0, logp_diff_t0), options=options)
        for i in range(len(plot_time) - 1):
            v_t = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[
                0]  # True_v(z_t0)
            g_t = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[
                1]

            z_t_samples.append(z_t1[i + 1].cpu().detach().numpy())
            g.append(g_t.cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())
            t_list.append(plot_time[i + 1])

        aa = 5  # 3
        angle1 = 10  # 30
        angle2 = 75  # 30
        trans = 0.8
        trans2 = 0.4
        widths = 0.2  # arrow width
        ratio1 = 0.4

        fig = plt.figure(figsize=(4 * 2, 3 * 2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 20

        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        # fig.suptitle(f'{t:.1f}day')
        ax1 = plt.axes(projection='3d')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.grid(False)
        ax1.set_xlabel('AE1')
        ax1.set_ylabel('AE2')
        ax1.set_zlabel('AE3')
        ax1.view_init(elev=angle1, azim=angle2)

        # 添加正方体后面的六根线
        ax1.plot([-2, -2], [2, 2], [-2, 2], color='gray', alpha=0.4,  linewidth=1.5)  # 左下后
        ax1.plot([-2, -2], [-2, -2], [-2, 2], color='gray', alpha=0.4,  linewidth=1.5)  # 左后下
        ax1.plot([-2, -2], [2, -2], [-2, -2], color='gray', alpha=0.4,  linewidth=1.5)  # 右下后
        ax1.plot([-2, 2], [-2, -2], [-2, -2], color='gray', alpha=0.4,  linewidth=1.5)  # 右后下
        ax1.plot([-2, -2], [-2, 2], [2, 2], color='gray', alpha=0.4,  linewidth=1.5)  # 左上后
        ax1.plot([-2, 2], [-2, -2], [2, 2], color='gray', alpha=0.4,  linewidth=1.5)  # 右下后


        ax1.plot([-2, 2], [2, 2], [-2, -2], color='black', alpha=1,  linewidth=1.5)  # 左下后
        ax1.plot([2, 2], [2, -2], [-2, -2], color='black', alpha=1,  linewidth=1.5)  # 左后下
        ax1.plot([2, 2], [-2, -2], [-2, 2], color='black', alpha=1,  linewidth=1.5)  # 右下后

        # ax1.spines['left'].set_linewidth(8.5)
        # ax1.spines['bottom'].set_linewidth(8.5)
        # ax1.spines['right'].set_linewidth(8.5)
        # ax1.spines['top'].set_linewidth(8.5)

        ax1.xaxis.set_pane_color((1, 1, 1, 0))
        ax1.yaxis.set_pane_color((1, 1, 1, 0))
        ax1.zaxis.set_pane_color((1, 1, 1, 0))



        ax1.invert_xaxis()
        ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))
        line_width = 0.3
        color_wanted = [np.array([0, 138, 69]) / 255,   #ipsc
                        np.array([225, 211, 115]) / 255,
                        np.array([175, 171, 171]) / 255,
                        np.array([197, 224, 180]) / 255,
                        np.array([250, 162, 25]) / 255,
                        np.array([125, 211, 247]) / 255,
                        np.array([178, 143, 234]) / 255,
                        np.array([243, 138, 134]) / 255,
                        ]
        # color_wanted = [np.array([0, 138, 69]) / 255,
        #                 np.array([225, 211, 115]) / 255,
        #                 np.array([179, 132, 186])/ 255,
        #                 np.array([70, 139, 202]) / 255,
        #                 np.array([242, 120, 115]) / 255,
        #                 ]
        # color_wanted = [np.array([0, 138, 69]) / 255,   #ipsc
        #                 np.array([225, 211, 115]) / 255,
        #                 np.array([179, 132, 186])/ 255,
        #                 np.array([70, 139, 202]) / 255,
        #                 np.array([250, 162, 25]) / 255,
        #                 np.array([204, 204, 204]) / 255,
        #                 np.array([125, 210, 246]) / 255,
        #                 np.array([255, 0, 0]) / 255,
        #                 ]


        # color_wanted = [np.array([242, 120, 115]) / 255,
        #                 np.array([225, 211, 115]) / 255,
        #                 np.array([0, 138, 69]) / 255,
        #                 np.array([70, 139, 202]) / 255,
        #                 np.array([179, 132, 186]) / 255,
        #                 np.array([3, 149, 198]) / 255,
        #                 np.array([180, 180, 213]) / 255,
        #                 np.array([178, 143, 237]) / 255]
        # color_wanted = [np.array([238, 68, 49]) / 255,
        #                 np.array([200, 150, 100]) / 255,
        #                 np.array([173, 219, 136]) / 255,
        #                 np.array([250, 187, 110]) / 255,
        #                 np.array([180, 180, 213]) / 255,
        #                 np.array([250, 199, 179]) / 255,
        #                 np.array([3, 149, 198]) / 255,
        #                 np.array([178, 143, 237]) / 255]
        for j in range(viz_samples):  # individual traj
            for i in range(len(plot_time) - 1):
                ax1.plot([z_t_samples[i][j, 0], z_t_samples[i + 1][j, 0]],
                         [z_t_samples[i][j, 1], z_t_samples[i + 1][j, 1]],
                         [z_t_samples[i][j, 2], z_t_samples[i + 1][j, 2]],
                         linewidth=0.5, color='grey', zorder=2)
        print(z_t_samples)
        # add inferrred trajecotry
        for i in range(len(sample_time)):
            ax1.scatter(z_t_samples[sample_time[i]][:, 0], z_t_samples[sample_time[i]][:, 1],
                        z_t_samples[sample_time[i]][:, 2], s=aa * 10, linewidth=0, color=color_wanted[i], zorder=3)
            ax1.quiver(z_t_samples[sample_time[i]][:, 0], z_t_samples[sample_time[i]][:, 1],
                       z_t_samples[sample_time[i]][:, 2],
                       v[sample_time[i]][:, 0] / v_scale, v[sample_time[i]][:, 1] / v_scale,
                       v[sample_time[i]][:, 2] / v_scale, color='k', alpha=1, linewidths=widths * 2,
                       arrow_length_ratio=0.12, zorder=4)

        # for i in range(len(integral_time)):
        #     ax1.scatter(z_t_data[i][:, 0], z_t_data[i][:, 1], z_t_data[i][:, 2], s=aa, linewidth=line_width, alpha=0.7,
        #                 facecolors='none', edgecolors=color_wanted[i], zorder=1)
        for i in range(0, len(integral_time), 3):  # 步长为3，每隔两个点绘制一个
            ax1.scatter(z_t_data[i][:, 0], z_t_data[i][:, 1], z_t_data[i][:, 2], s=aa, linewidth=line_width, alpha=0.7,
                        facecolors='none', edgecolors=color_wanted[i], zorder=1)
        plt.savefig(os.path.join(args.save_dir, f"traj_3d.pdf"), format="pdf", pad_inches=0.1, bbox_inches='tight')
        plt.show()