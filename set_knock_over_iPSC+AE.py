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





# 定义要操作的基因列表
#top = ["EPCAM", "MIXL1", "HAND1", "EOMES", "PTCH1", "GATA6", "GSC", "DKK1", "GATA4", "HNFA4"]
#_20_knockout   _50_knockout _100_knockout  _20_overexpress _50_overexpress _100_overexpress
situ='_50_overexpress_分时'
in_path = f"D:\\R111\\R\\R_use\\新数据3\\normalized_data{situ}.csv"
data = pd.read_csv(in_path)

name1=f"normalized_data{situ}.csv"

def load_data(dataset:str,path_to_data):
    filename = path_to_data+name1
    df = pd.read_csv(filename, header=[0, 1], index_col=0)
    times = df.columns.get_level_values(0)
    times = times.to_list()
    df.columns = df.columns.droplevel(0)
    df = df.transpose()
    print(df)
    adata = sc.AnnData(df)
    adata.obs['time'] = times
    X=adata.X

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
    folder=Path('results/'+dataset+'_'+name1+'_'+str(seed)+\
           '_'+str(n_latent)+'_'+str(n_layers)+'_'+str(n_hidden)+\
           '_'+str(dropout)+'_'+str(weight_decay)+'_'+str(lr)+'_'+str(batch_size)+'/')
    return folder





# 如果列名正确，确保在 generate_plots 函数中使用正确的列名
def generate_plots(folder, model, adata, seed, n_neighbors=10, min_dist=0.5, plots='umap'):
    model.eval()
    with torch.no_grad():
        X_latent_AE = model.get_latent_representation(torch.tensor(adata.X).type(torch.float32).to('cpu'))
    adata.obsm['X_AE'] = X_latent_AE.detach().cpu().numpy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_AE')

    # 使用正确的列名作为颜色参数
    color = adata.obs.columns.tolist()  # 假设列名就是颜色参数

    if plots == 'umap':
        sc.tl.umap(adata, random_state=seed, min_dist=min_dist)
        with rc_context({'figure.figsize': (8, 8 * len(color))}):
            sc.pl.umap(adata, color=color,
                       legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2)
        plt.savefig(str(folder) + '/umap.pdf')
        plt.close()
    elif plots == 'embedding':
        with rc_context({'figure.figsize': (8 * len(color), 8)}):
            sc.pl.embedding(adata, 'X_AE', color=color,
                            legend_loc='on data',
                            legend_fontsize=12,
                            legend_fontoutline=2)
            plt.legend(frameon=False)
            plt.xticks([plt.xlim()[0], 0., plt.xlim()[1]])
            plt.yticks([plt.ylim()[0], 0., plt.ylim()[1]])
        plt.savefig(str(folder) + '/embedding.pdf')
        plt.close()
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
         path_to_data='D:\\R111\\R\\R_use\\新数据3\\1\\'
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

def  savedata(folder, adata,seed):
    with torch.no_grad():
        X_latent_AE=model.get_latent_representation(torch.tensor(adata.X).type(torch.float32).to('cpu'))
    adata.obsm['X_AE']=X_latent_AE.detach().cpu().numpy()
    X_AE = adata.obsm['X_AE']
    # 提取时间信息
    time_info = adata.obs['time'].to_numpy()
    print(time_info)
    # 创建 DataFrame
    results_df = pd.DataFrame({
        f'X_AE{1}': X_AE[:, 0],
        f'X_AE{2}': X_AE[:, 1],
        f'X_AE{3}': X_AE[:, 2],
        'Time': time_info
    })

    # 保存到 CSV 文件
    csv_filename = f'umap_results_seed_{seed}.csv'
    results_df.to_csv(os.path.join(folder, csv_filename), index=False)
    return results_df

# 以下是调用main函数并进行模型训练和绘图的代码
seed=4234
n_layers = 2
batch_size=256
dataset=f'normalized{situ}'

lr=1e-3
n_hidden=400
n_latent = 10

model,trainer, adata,folder=main(dataset=dataset,seed=seed,
                          n_layers=n_layers,n_latent=n_latent,n_hidden=n_hidden,
                          activation='relu',
                          lr=lr,batch_size=batch_size,
                          max_epoch=500,
                          mode='training',
                          path_to_data = 'D:\\R111\\R\\R_use\\新数据3\\')

model=model.to('cpu')

# 检查 adata.obs 中的列名
print(adata.obs.columns)
# generate_plots(folder,model, adata,seed,n_neighbors=20,min_dist=0.5,plots='embedding')
generate_plots(folder,model, adata,seed,n_neighbors=30,min_dist=1.5,plots='umap')
