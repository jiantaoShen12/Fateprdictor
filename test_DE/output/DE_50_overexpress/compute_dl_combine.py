#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
np.random.seed(0)


path_in="/home/sjt/workspace/beginning_project/cell_change尝试/test_iPSC/output/iPSC_10_knockout/LSTM/"
df = pd.read_csv(path_in+f'df_dl.csv')
df1=df[df['classifier'] == 'c1']
df1.dropna(axis=0, how='any')
size1 = df1.iloc[:, :6]
weighted_preds1 = np.zeros(size1.shape)
weighted_preds2 = np.zeros(size1.shape)
pseudotime_index_array = df1['time'].values

for a in range(1,4):
# Load models
    if  a == 1:
        path_in="/home/sjt/workspace/beginning_project/cell_change尝试/test_iPSC/output/iPSC_10_knockout/LSTM/"
        df = pd.read_csv(path_in+f'df_dl.csv')
        df1=df[df['classifier'] == 'c1']
        df1.dropna(axis=0, how='any')
        df2=df[df['classifier'] == 'c2']
        df1.dropna(axis=0, how='any')
        print(df1)
        weights_1 = [3/15, 3/15, 7/15,7/15, 7/15,5/15]
        weights_2 = [3/15, 3/15, 7/15,3/15, 6/15,3/15]
        size1 = df2.iloc[:, :6]
        weighted_preds = np.zeros(size1.shape)
    elif a == 2:
        path_in="/home/sjt/workspace/beginning_project/cell_change尝试/test_iPSC/output/iPSC_10_knockout/GRU/"
        df = pd.read_csv(path_in+f'df_dl.csv')
        df1=df[df['classifier'] == 'c1']
        df1.dropna(axis=0, how='any')
        df2=df[df['classifier'] == 'c2']
        df1.dropna(axis=0, how='any')
        weights_1=  [5/15, 5/15, 4/15,5/15, 4/15,5/15]
        weights_2 = [7/15, 7/15, 4/15,5/15, 3/15,6/15]

    elif a == 3:
        path_in="/home/sjt/workspace/beginning_project/cell_change尝试/test_iPSC/output/iPSC_10_knockout/BILSTM/"
        df = pd.read_csv(path_in+f'df_dl.csv')
        df1=df[df['classifier'] == 'c1']
        df1.dropna(axis=0, how='any')
        df2=df[df['classifier'] == 'c2']
        df1.dropna(axis=0, how='any')
        weights_1= [7/15, 7/15, 4/15,3/15, 4/15,5/15]
        weights_2 = [5/15, 5/15, 4/15,7/15, 6/15,6/15]

    # 使用iloc来选取前六列
    preds_prob_1 = df1.loc[:, ['0','1', '2', '3', '4', '5']]
    preds_prob_1.dropna(axis=0, how='all')
    preds_prob_2 = df1.loc[:, ['0','1', '2', '3', '4', '5']]
    preds_prob_2.dropna(axis=0, how='all')
    print(preds_prob_1)

    # 将预测结果的第j列乘以权重
    for j in range(len(weights_1)):
        weighted_preds1[:, j] += preds_prob_1.iloc[:, j] * weights_1[j]
    print(weighted_preds1)

    for j in range(len(weights_2)):
        weighted_preds2[:, j] += preds_prob_2.iloc[:, j] * weights_2[j]

preds_prob=weighted_preds1+weighted_preds2
preds_prob = preds_prob / preds_prob.sum(axis=1, keepdims=True)



name1=['Null', 'Fold', 'Ts', 'Pf', 'Pd', 'Ns']
preds_prob_df = pd.DataFrame(preds_prob, columns=name1)

# 创建一个图形
fig = go.Figure()
pseudotime=df1['time'].values

print(preds_prob_df)
print(pseudotime)
# 为每一列添加一个折线图轨迹
for idx, column in enumerate(preds_prob_df.columns):
    # 选择transition之前的点进行绘制
    trace_x = pseudotime
    trace_y = preds_prob_df[column]

    fig.add_trace(
        go.Scatter(
            x=trace_x,  # 使用transition之前的x坐标
            y=trace_y,  # 使用transition之前对应的概率值
            name=name1[idx],  # 使用类别标签作为图例名称
            mode='lines',
        )
    )

# 更新图形的布局
fig.update_layout(
    title='Deep Learning Probability Distribution Over Pseudotime',
    xaxis_title='Pseudotime',
    yaxis_title='Probability',
    #xaxis_range=[0,500],
    showlegend=True,
    legend_traceorder='normal'
)

# 显示图形
fig.show()


# 设置图形的保存路径
file_path = 'sjt/workspace/beginning_project/cell_change尝试/test_iPSC/output/iPSC_10_knockout/combined_pd.png'
# 保存图形到文件
pio.write_image(fig, file_path)
# 显示保存成功的消息
print(f'Figure saved to {file_path}')