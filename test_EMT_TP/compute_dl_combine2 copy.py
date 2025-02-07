#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
np.random.seed(0)

# 读取数据
path_in = "/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/对比/"
df = pd.read_csv(path_in + 'df_dl_null_fixed.csv')
df1 = df[df['classifier'] == 'm1']
df2 = df[df['classifier'] == 'm2']

# 选择前6列

# 找到两个DataFrame的共同索引
size1 = df1.iloc[:, :6].reset_index(drop=True)
size2 = df2.iloc[:, :6].reset_index(drop=True)


# 相加
preds_prob = size1.add(size2, fill_value=0)

# 归一化
row_sums = preds_prob.sum(axis=1)
preds_prob = preds_prob.div(row_sums, axis=0)

# 选择其他列
size3 = df2.iloc[:, 6:7].reset_index(drop=True)
size4 = df2.iloc[:, 9].reset_index(drop=True)

# 合并
dl = pd.concat([preds_prob, size3, size4], axis=1)
dl.to_csv('sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/对比1/df_dl_null_fixed.csv', index=False)

# 重复上述步骤处理 df_dl_fixed.csv
df = pd.read_csv(path_in + 'df_dl_fixed.csv')
df1 = df[df['classifier'] == 'm1']
df2 = df[df['classifier'] == 'm2']

# 选择前6列

# 找到两个DataFrame的共同索引
size1 = df1.iloc[:, :6].reset_index(drop=True)
size2 = df2.iloc[:, :6].reset_index(drop=True)


# 相加
preds_prob = size1.add(size2, fill_value=0)

# 归一化
row_sums = preds_prob.sum(axis=1)
preds_prob = preds_prob.div(row_sums, axis=0)

# 选择其他列
size3 = df2.iloc[:, 6:7].reset_index(drop=True)
size4 = df2.iloc[:, 9].reset_index(drop=True)

# 合并
dl = pd.concat([preds_prob, size3, size4], axis=1)
dl.to_csv('sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/对比1/df_dl_fixed.csv', index=False)