import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utility import *
'''
这段代码是一个深度学习模型训练流程的实现，其最终目标是训练一个名为 `UOT` 的神经网络模型，并通过时间序列数据来拟合一个动态系统。具体来说，代码执行完以下任务：

1. **模型训练**：通过迭代优化过程，调整模型参数以最小化损失函数。
2. **损失计算**：在每次迭代中，计算并记录模型的损失值，以监控训练进度。
3. **学习率调整**：使用预定的策略（如分段恒定衰减）调整学习率。
4. **模型保存**：在训练过程中定期保存模型的状态，以及在训练结束时保存最终状态。
5. **结果记录**：记录训练过程中的关键信息，如损失值、传输成本、L2 范数值和正态分布的sigma值。
最终，代码得到以下结果：
- **训练好的模型**：经过多次迭代后，模型的参数被优化，以适应训练数据。
- **损失曲线**：记录了训练过程中损失函数的变化，可用于分析模型的收敛情况。
- **传输成本**：如果模型需要评估传输成本效率，该值也会被记录。
- **L2 范数值**：记录了模型预测和实际数据之间差异的度量。
- **正态分布的sigma值**：如果代码中使用了基于高斯噪声的数据增强或正态分布假设，sigma值的变化会被记录。
此外，代码中还包含了可视化函数（如`plot_3d`, `plot_jac_v`, `plot_grad_g`）的调用
，这些函数可以生成模型预测的轨迹、雅可比矩阵和梯度的图表，帮助研究者更好地理解模型的行为和性能。
最后，所有这些结果（模型参数、优化器状态、损失值等）以及可视化图表都被保存到指定的目录中，通常是一个名为 `ckpt.pth` 的文件
，以及可能的其它中间检查点文件。这些保存的文件可以用于后续的模型评估、分析或重新训练。
'''

    
    
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
    func.apply(initialize_weights)


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            
            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            
            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())
            
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
            
            if itr % 500 == 0:
                ckpt_path = os.path.join(args.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                
            
            

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))


    
    
    
    