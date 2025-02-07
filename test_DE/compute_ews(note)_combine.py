#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:52:37 2022
Compute early warning signals in gene data of first PCA component
这段代码的作用是计算第一主成分分析（PCA）成分的基因数据中的早期预警信号（EWS），用于探测细胞状态的潜在临界转变。具体步骤包括：
"D:\pythonku\anaconda3\envs\fatenet\Lib\site-packages\tensorflow\python\tf2.py"  在这里libiomp5md
1. **导入库**：代码开始部分导入了NumPy、Pandas、Matplotlib、Plotly、ewstools、TensorFlow等库。
2. **设置随机种子**：使用`np.random.seed(6)`确保结果的可重复性。
3. **读取数据**：读取存储在CSV文件中的PCA数据。
4. **加载深度学习模型**：加载之前训练好的两个深度学习分类器模型。
5. **选择特定聚类和PCA成分**：选择特定聚类（cluster 5）和PCA成分（pca_comp 0）的数据进行分析。
6. **计算EWS**：
   - 使用ewstools库来处理时间序列数据，包括去趋势、计算滚动方差和自相关。
   - 应用深度学习分类器来预测转变。
   - 生成EWS的Plotly图表并保存为HTML文件。
7. **添加伪时间列**：将原始数据中的伪时间信息添加到DL预测和EWS状态数据中。
8. **导出数据**：将DL预测结果和EWS状态数据导出为CSV文件。
9. **计算空模型的EWS**：通过从原始时间序列的残差中随机采样并添加到趋势中，来模拟没有实际转变的情况，并计算这种情况下的EWS。
10. **导出空模型的EWS数据**：将空模型的EWS结果导出为CSV文件。
这段代码对应于论文中的实验部分，特别是与分析细胞状态转变和预测临界转变相关的部分。
在论文中，作者可能使用这些计算出的EWS来展示其方法如何能够探测和预测细胞状态的转变，例如从未分化状态到特定细胞类型的转变。
通过这种方法，研究者可以在系统发生临界转变之前提供早期预警，这对于理解细胞命运决策和可能的医疗干预至关重要。
@author: tbury
"""

import numpy as np
import pandas as pd
import ewstools
import tensorflow as tf
import os

tf.compat.v1.logging.set_verbosity(  tf.compat.v1.logging.ERROR)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model
#参数选择

# def zscore(s, window, thresh=2, return_all=False):
#     roll = s.rolling(window=window, min_periods=1, center=True)
#     avg = roll.mean()
#     std = roll.std(ddof=0)
#     z = s.sub(avg).div(std)
#     m = z.between(-thresh, thresh)

#     if return_all:
#     return z, avg, std, m
#     return s.where(m, avg)

np.random.seed(6)
T=['DE','DE_20_overexpress','DE_25_overexpress','DE_50_overexpress','DE_25_knockout','DE_50_knockout','DE_200_knockout']
p=6
# Import PCA data
df = pd.read_csv(f"/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/{T[p]}/trajectory_and_pseudotime.csv")
# 检查DataFrame是否为空
if df.empty:
    print("DataFrame is empty. Please check the file path and content.")
else:
    print("DataFrame is not empty and contains", len(df), "rows.")
inc =10
span=0.2
start_index =0
df_select=df
transition=260
#92 194 260 432 571
start_idx = max(transition - 499, 0)
df_select = df_select.iloc[start_idx:]
s=df_select["X_AE1"].copy()
print(df_select)

for a in range(1,4):
    if  a == 1:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        c2 = load_model(path)
        output_dir=f"/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/{T[p]}/LSTM/"
    elif a == 2:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        c2 = load_model(path)
        output_dir=f"/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/{T[p]}/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        c2 = load_model(path)
        output_dir=f"/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/{T[p]}/BILSTM/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Compute EWS
    #print("Compute EWS for PCA comp {}".format(pca_comp))
    # time series increment for DL classifier
    #ts = ewstools.TimeSeries(s,transition=transition)
    ts = ewstools.TimeSeries(s,transition=transition)
    ts.detrend(method="Lowess", span=span)
    # ts.detrend(method='Gaussian', bandwidth=span)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)

    ts.apply_classifier_inc(c1, inc=inc, name="c1", verbose=0)
    ts.apply_classifier_inc(c2, inc=inc, name="c2", verbose=0)

    # Make quick fig of EWS
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    #fig.update_layout(title="PCA comp={}".format(pca_comp))
    fig.write_html(f"{output_dir}temp.html")

    # Export data
    df_dl_forced = ts.dl_preds
    df_ews_forced = ts.state
    df_ews_forced = df_ews_forced.join(ts.ews)
    print(df_dl_forced.columns)
    #merged_df = pd.merge(df_dl_forced, df_select[['pseudotime1','pseudotime']], on='time', how='inner')
    df_dl_forced.to_csv(f"{output_dir}df_dl.csv", index=False)
    #merged_df = pd.merge(df_ews_forced, df_select[['pseudotime1','pseudotime']], on='time', how='inner')
    df_ews_forced.to_csv(f"{output_dir}df_ews.csv")
