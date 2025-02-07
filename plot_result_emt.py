import os
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utility import *

h='ckpt.pth'

#EMT  Output/EMT   ckpt  5



if __name__ == '__main__':
    args=create_args()
    torch.enable_grad()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu)
                            if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(args,device)
    integral_time = args.timepoints

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]

    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)


    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, f'{h}')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
            func.load_state_dict(checkpoint['func_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))


    def Sampling1(num_samples, time_all, time_pt, data_train, sigma, device):
        # perturb the  coordinate x with Gaussian noise N (0, sigma*I )
        rg = random.Random()  # 创建一个独立的随机数生成器实例
        rg.seed(0)  # 只为这个实例设置种子
        mu = data_train[time_all[time_pt]]
        print(data_train)
        print(mu)
        num_gaussian = mu.shape[0]  # mu is number_sample * dimension
        print(num_gaussian.imag)
        print(mu)
        dim = mu.shape[1]
        sigma_matrix = sigma * torch.eye(dim)
        # check if number of points is <num_samples
        if num_gaussian < num_samples:
            samples = mu[rg.choices(range(0, num_gaussian), k=num_samples)]
        else:
            samples = mu[rg.sample(range(0, num_gaussian), num_samples)]
        return samples


    # plot 3d of inferred trajectory of 20 cells
    def plot_3d11(func, data_train, train_time, integral_time, args, device):
        #初始化变量

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

#使用Torch进行推断

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
    #设置ODE求解器选项
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
    #使用 ODE 求解器推断轨迹

            z_t1, _, logp_diff_t1 = odesolve(func, y0=(z_t0, g0, logp_diff_t0), options=options)
            for i in range(len(plot_time) - 1):
                v_t = \
                func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[
                    0]  # True_v(z_t0)
                g_t = \
                func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_t1[i + 1], g0, logp_diff_t1))[1]

                z_t_samples.append(z_t1[i + 1].cpu().detach().numpy())
                g.append(g_t.cpu().detach().numpy())
                v.append(v_t.cpu().detach().numpy())
                t_list.append(plot_time[i + 1])
    #可视化推断的轨迹

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
            v_scale = 5

            plt.tight_layout()
            plt.axis('off')
            plt.margins(0, 0)
            # fig.suptitle(f'{t:.1f}day')
            ax1 = plt.axes(projection='3d')
            ax1.grid(False)
            ax1.set_xlabel('AE1')
            ax1.set_ylabel('AE2')
            ax1.set_zlabel('AE3')
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)
            ax1.set_zlim(-2, 2)
            ax1.set_xticks([-2, 0,2])
            ax1.set_yticks([-2, 0, 2])
            ax1.set_zticks([-2, 0, 2])
            ax1.view_init(elev=angle1, azim=angle2)

            ax1.xaxis.set_pane_color((1, 1, 1, 0))
            ax1.yaxis.set_pane_color((1, 1, 1, 0))
            ax1.zaxis.set_pane_color((1, 1, 1, 0))

            ax1.invert_xaxis()
            ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))
            line_width = 0.3

            color_wanted = [np.array([0, 138, 69]) / 255,
                            np.array([225, 211, 115]) / 255,
                            np.array([179, 132, 186]) / 255,
                            np.array([70, 139, 202]) / 255,
                            np.array([242, 120, 115]) / 255,
                            ]
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

            # for i in range(len(integral_time)):
            #     ax1.scatter(z_t_data[i][:, 0], z_t_data[i][:, 1], z_t_data[i][:, 2], s=aa, linewidth=line_width,
            #                 alpha=0.7, facecolors='none', edgecolors=color_wanted[i], zorder=1)
            for i in range(0, len(integral_time), 2):  # 步长为3，每隔两个点绘制一个
                ax1.scatter(z_t_data[i][:, 0], z_t_data[i][:, 1], z_t_data[i][:, 2], s=aa, linewidth=line_width,
                            alpha=0.7,
                            facecolors='none', edgecolors=color_wanted[i], zorder=1)            # plt.savefig(os.path.join(args.save_dir, f"traj_3d.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
            plt.show()

        # 遍历每个时间点的数据
        all_data = []

        for time_index, time in enumerate(integral_time):
            # 获取当前时间点的所有特征数据
            # 假设data_train[time_index]是一个二维张量，其中行是样本，列是特征
            for sample_index in range(z_t_samples[time_index].shape[0]):
                # 提取当前样本的所有特征
                features = z_t_samples[time_index][sample_index].tolist()
                # 将伪时间和特征数据添加到列表中
                all_data.append([time] + features)

        # 创建DataFrame
        # 假设特征数量是固定的，我们可以根据data_train的第一个元素来获取特征名称
        feature_names = [f'Feature{i + 1}' for i in range(len(z_t_samples[0][0]))]
        df = pd.DataFrame(all_data, columns=['Pseudotime'] + feature_names)




    # generate the plot of trajecotry

    plot_3d1(func,data_train,train_time,integral_time,args,device)
    # Average Jacobian matrix of cells at day 0
    plot_jac_v(func,data_train[1],1,'Average_jac_d0.pdf',['AE1','AE2','AE3'],args,device)
    # Average gradients of growth rate of cells at day 0
    plot_grad_g(func,data_train[1],1,'Average_grad_d0.pdf',['AE1','AE2','AE3'],args,device)
