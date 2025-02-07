import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import scanpy as sc
from matplotlib.pyplot import rc_context
import os
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import sys
sys.path.append('Path to AE folder')
from AE import AutoEncoder, Trainer
dataset='EMT'
# 这段代码是一个Python脚本，用于处理和分析生物信息学数据，特别是涉及基因表达数据。它使用了多个Python库，包括NumPy、PyTorch、Pandas、Scanpy和Matplotlib。以下是代码的主要功能和步骤的概述：
#
# 1. **导入库**：代码开始部分导入了所需的Python库。
#
# 2. **设置变量**：定义了一些变量，如`dataset`和`h`，这些变量用于控制数据处理的逻辑。
#
# 3. **定义要删除的列名列表**：`top`变量包含了一些基因名称，这些基因在后续的数据处理中会被特别处理。
#
# 4. **读取数据**：使用Pandas读取CSV文件中的数据。
#
# 5. **数据处理**：根据`h`变量的值，对数据进行不同的处理，例如将特定基因的表达量设置为0或翻倍。
#
# 6. **定义函数**：
#    - `load_data`：加载数据并创建AnnData对象，这是Scanpy库中用于存储和处理单细胞RNA-seq数据的对象。
#    - `folder_dir`：创建一个包含多个参数的文件夹路径。
#    - `generate_plots`：生成降维图（如UMAP）并保存。
#    - `savedata`：保存处理后的数据到CSV文件。
#    - `loss_plots`：绘制并保存模型损失图。
#    - `main`：主函数，用于训练自动编码器模型，并根据模式（训练或加载）执行不同的操作。
#
# 7. **模型训练和结果保存**：使用`main`函数训练模型，生成图表，保存模型和结果。
#
# 8. **数据归一化和保存**：使用`MinMaxScaler`对数据进行归一化处理，并将不同时间点的数据保存为NumPy数组。
#
# 代码中有一些重复的变量赋值（例如`h='25_overexpress'`出现了三次），这可能是不必要的，除非有特定的逻辑需要这样做。此外，代码中的注释和变量命名表明它是用于特定生物信息学分析的，特别是与EMT（上皮-间充质转化）相关的数据。
#
# 如果你有任何具体的问题或需要进一步的解释，请告诉我。

h='25_overexpress_no归一'
h='25_overexpress'
h='25_overexpress'
h='null'
h='25_overexpress'
h='25_konckout'
# 定义要删除的列名列表
h='25_null'

top=["LCE3D"   "MT2A"    "GPX2"    "AKR1B10" "SPINK6" "IGFBP7"  "IGFL1"   "TGFBI"   "TAGLN"   "KRT81"]
top = ["RPS4Y1", "ID3", "PMEPA1", "IGFBP4", "AKR1C3", "JUNB", "COTL1", "NEAT1", "SAT1", "ID1",
       "PTTG1", "CKS2", "KIAA0101", "TUBB4B", "TP53I3", "CAV1", "PPP1R14A", "FTL", "TPM1",
       "SERPINE1", "NCL", "OCIAD2", "C12orf75", "SLC3A2", "CKS1B"
       ]
# 读取数据
in_path = "D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\AE_EMT_normalized.csv"
data = pd.read_csv(in_path)
if  'konckout'in h:
    for index, row in data.iterrows():
        if row.iloc[0] in top:
            data.loc[index, data.columns[1:]] = 0
    data.to_csv(f'D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\AE_EMT_{h}.csv', index=False)
    name1=f'AE_EMT_{h}'
elif  'overexpress'in h:
    # 将指定的列设置为0
    for index, row in data.iterrows():
        if row.iloc[0] in top:
            data.loc[index, data.columns[1:]] = data.loc[index, data.columns[1:]] * 2
    data.to_csv(f'D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\AE_EMT_{h}.csv', index=False)
    name1=f'AE_EMT_{h}'

elif  'null'in h:
    data.to_csv(f'D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\AE_EMT.csv', index=False)
    name1=f'AE_EMT'

def load_data(ataset:str,path_to_data):

    if dataset=='EMT':
        data = pd.read_csv(path_to_data+name1+'.csv', index_col=0).transpose()
        y = pd.read_csv(path_to_data+'AE_EMT_time.csv', index_col=0)
        row_order = data.index
        y_reordered = y.loc[row_order]
        adata = sc.AnnData(data)
        adata.obs['time'] = y_reordered


        X=adata.X
    elif dataset=='iPSC':
        filename = path_to_data+'data.xlsx'
        df = pd.read_excel(filename, header=[0, 1], index_col=0)
        times = df.columns.get_level_values(0)
        times = times.to_list()
        df.columns = df.columns.droplevel(0)
        df = df.transpose()
        adata = sc.AnnData(df)
        adata.obs['time'] = times

        X=adata.X

    else:
        raise NotImplementedError
    return adata,X
def folder_dir(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str = 'relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         batch_size: int=32,):
    folder=Path('results/'+name1+'_'+str(seed)+\
           '_'+str(n_latent)+'_'+str(n_layers)+'_'+str(n_hidden)+\
           '_'+str(dropout)+'_'+str(weight_decay)+'_'+str(lr)+'_'+str(batch_size)+'/')
    return folder
def generate_plots(folder,model, adata,seed,n_neighbors=10,min_dist=0.5,plots='umap'):
    model.eval()
    with torch.no_grad():
        X_latent_AE=model.get_latent_representation(torch.tensor(adata.X).type(torch.float32).to('cpu'))
    adata.obsm['X_AE']=X_latent_AE.detach().cpu().numpy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors,use_rep='X_AE')
    if dataset in ['EMT','iPSC']:
        color=['time']
    else:
        raise  NotImplementedError
    if plots=='umap':
        sc.tl.umap(adata,random_state=seed,min_dist=min_dist)
        with rc_context({'figure.figsize': (8, 8*len(color))}):
            sc.pl.umap(adata, color=color,
                       legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
        plt.savefig(str(folder) + '/umap.pdf')
        plt.close()
    elif plots=='embedding':
        with rc_context({'figure.figsize': (8*len(color), 8)}):
            sc.pl.embedding(adata, 'X_AE',color=color,
                       # legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
            plt.legend(frameon=False)
            plt.xticks([plt.xlim()[0], 0., plt.xlim()[1]])
            plt.yticks([plt.ylim()[0], 0., plt.ylim()[1]])
        plt.savefig(str(folder) + '/embedding.pdf')
        plt.close()
def  savedata(folder, adata,seed):
    X_AE = adata.obsm['X_AE']
    # 提取时间信息
    time_info = adata.obs['time'].to_numpy()

    # 创建 DataFrame
    results_df = pd.DataFrame({
        f'X_AE1': X_AE[:, 0],
        f'X_AE2': X_AE[:, 1],
        f'X_AE3': X_AE[:, 2],

        'Time': time_info
    })
    '''
            f'X_AE4': X_AE[:, 3],
            f'X_AE5': X_AE[:, 4],
            f'X_AE6': X_AE[:, 5],
            f'X_AE7': X_AE[:, 6],
            f'X_AE8': X_AE[:, 7],
            f'X_AE9': X_AE[:, 8],
            f'X_AE10': X_AE[:,9],
    '''
    # 保存到 CSV 文件
    csv_filename = f'umap_results_seed_{seed}.csv'
    results_df.to_csv(os.path.join(folder, csv_filename), index=False)
    return results_df
def loss_plots(folder,model):
    fig,axs=plt.subplots(1, 1, figsize=(4, 4))
    axs.set_title('AE loss')
    axs.plot(model.history['epoch'], model.history['train_loss'])
    axs.plot(model.history['epoch'], model.history['val_loss'])
    plt.yscale('log')
    axs.legend(['train loss','val loss'])
    plt.savefig(str(folder)+'/loss.pdf')
    plt.close()

def main(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str='relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         max_epoch:int=500,
         batch_size: int=32,
         mode='training',
         path_to_data='D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\'
         ):
    adata,X = load_data(dataset,path_to_data)

    model=AutoEncoder(in_dim=X.shape[1],
                      n_latent=n_latent,
                      n_hidden=n_hidden,
                      n_layers=n_layers,
                      activate_type=activation,
                      dropout=dropout,
                      norm=True,
                      seed=seed,)

    trainer=Trainer(model,X=X,
                    test_size=0.1,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    seed=seed)

    folder=folder_dir(dataset=dataset,
         seed=seed,
         n_latent=n_latent,
         n_hidden=n_hidden,
         n_layers=n_layers,
         dropout=dropout,
         activation=activation,
         weight_decay=weight_decay,
         lr=lr,
         batch_size=batch_size,)

    if mode=='training':
        print('training the model')
        trainer.train(max_epoch=max_epoch,patient=30)

        # model.eval()
        if not os.path.exists(folder):
            folder.mkdir(parents=True)
        torch.save({
            'func_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss_history':trainer.model.history,
        }, os.path.join(folder,'model.pt'))
    elif mode=='loading':
        print('loading the model')
        check_pt = torch.load(os.path.join(folder, 'model.pt'))

        model.load_state_dict(check_pt['func_state_dict'])
        trainer.optimizer.load_state_dict(check_pt['optimizer_state_dict'])
        model.history=check_pt['loss_history']
    return model,trainer,adata,folder



seed=4232
n_layers = 1
batch_size=128
lr=1e-3
n_hidden=300
n_latent = 10


model,trainer, adata,folder=main(dataset=dataset,seed=seed,
                          n_layers=n_layers,n_latent=n_latent,n_hidden=n_hidden,
                          activation='relu',
                          lr=lr,batch_size=batch_size,
                          max_epoch=500,
                          #mode='loading',
                          path_to_data = 'D:\\python2\\store\\c\\cellseries\\time\\EMT_data\\')

#model=model.to('cpu')
generate_plots(folder,model, adata,seed,n_neighbors=20,min_dist=0.5,plots='embedding')
generate_plots(folder,model, adata,seed,n_neighbors=20,min_dist=0.5,plots='umap')
loss_plots(folder,model)
print(adata.X)
print(adata.obs)
print(adata.var)
print(adata.obsm)
print(adata.varm)
print(adata.uns)
data=savedata(folder, adata,seed)

n=3#维度
time_list = ['0d', '8h', '1d', '3d', '7d']
from sklearn.preprocessing import MinMaxScaler
# 初始化 MinMaxScaler，设置归一化的范围为 -2 到 2
scaler = MinMaxScaler(feature_range=(-2, 2))
normalized_data = data.iloc[:,0:n]
normalized_data=scaler.fit_transform(normalized_data)
data.iloc[:,0:n]=normalized_data[:,0:n]
# 创建一个对象数组，用于存储不同时间点的数据
data_all = np.empty((1, 5), dtype=object)

# 循环处理每个时间点的数据
for i in range(0, 5):
    h = data[data['Time'] == time_list[i]].iloc[:, 0:n]
    npdata = h.to_numpy()
    # 获取数组的形状
    rows, cols = npdata.shape
    renpdata = npdata.reshape(rows, cols)

    # 将数据存储到对象数组中
    data_all[0, i] = renpdata

# 指定保存路径
npy_file_path = f'D:\\python2\\store\\c\\cellseries\\time\\input\\{name1}_3wei.npy'
# 保存数组到.npy文件
np.save(npy_file_path, data_all)
print(f"数据已保存到 {npy_file_path}")