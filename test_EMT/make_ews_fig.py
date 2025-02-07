#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
np.random.seed(0)

    # "state": "rosybrown",
    # "smoothing": "red",
    # "variance": cols[1],
    # "ac": cols[2],
len1=129

T=2.5; T1=0.5 ;T2=2.5
T=4.0; T1=2 ;T2=4

T=4.5; T1=2.5 ;T2=4.5
T=2.0; T1=0 ;T2=2
T=2.5; T1=0.5 ;T2=2.5
T=4.0; T1=2 ;T2=4
T=2.5; T1=0.5 ;T2=2.5

file_path = f'/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT/output/{T1}_{T2}'
df = pd.read_csv(f'/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT/output/{T}/LSTM/df_ews.csv')
state_df = pd.DataFrame(df)

# 创建一个图形对象
fig = go.Figure()

# 绘制状态变量
fig.add_trace(
    go.Scatter(
        x=state_df.time.values,
        y=state_df["state"].values,
        name="state",
        marker_color='grey',  # 指定轨迹的颜色
    )
)

# 如果存在平滑变量，也绘制它
if "smoothing" in state_df.columns:
    fig.add_trace(
        go.Scatter(
            x=state_df.time.values,
            y=state_df["smoothing"].values,
            name="smoothing",
            line_color='navy',  # 指定轨迹的颜色
            line_width=3.5 # 设置轨迹的宽度为3
        )
    )
# 保存图形到文件

fig.update_layout(
    xaxis_range=[int(T1*len1), int(T2*len1)],  # 设置x轴范围为完整数据集的范围
    showlegend=True,  # 设置为False以隐藏图例    legend_traceorder='normal',
    plot_bgcolor='white',  # 设置图形背景颜色为白色
    xaxis=dict(
        gridcolor='white',  # 设置x轴背景颜色为白色
        color='black',  # 设置x轴刻度和标签颜色为黑色
        linecolor='black',  # 设置x轴线条颜色为黑色
        tickcolor='black',  # 设置x轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线        # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2
    ),
    yaxis=dict(
        gridcolor='white',  # 设置y轴背景颜色为白色
        color='black',  # 设置y轴刻度和标签颜色为黑色
        linecolor='black',  # 设置y轴线条颜色为黑色
        tickcolor='black',  # 设置y轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线       # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2
    )
)
pio.write_image(fig, file_path+'_ews.png')
# 显示保存成功的消息
print(f'Figure1 saved to {file_path}')
#fig.show()


# 创建一个图形对象
fig1 = go.Figure()

if "variance" in state_df.columns:
    fig1.add_trace(
        go.Scatter(
            x=state_df.time.values,
            y=state_df["variance"].values,
            name="Variance",
            line_color='olive',  # 指定轨迹的颜色
            line_width=3.5 # 设置轨迹的宽度为3
        )
    )

# 更新图形的布局
fig1.update_layout(
    xaxis_range=[int(T1*len1), int(T2*len1)],  # 设置x轴范围为完整数据集的范围
    showlegend=True,  # 设置为False以隐藏图例    legend_traceorder='normal',
    plot_bgcolor='white',  # 设置图形背景颜色为白色
    xaxis=dict(
        gridcolor='white',  # 设置x轴背景颜色为白色
        color='black',  # 设置x轴刻度和标签颜色为黑色
        linecolor='black',  # 设置x轴线条颜色为黑色
        tickcolor='black',  # 设置x轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线        # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2
    ),
    yaxis=dict(
        gridcolor='white',  # 设置y轴背景颜色为白色
        color='black',  # 设置y轴刻度和标签颜色为黑色
        linecolor='black',  # 设置y轴线条颜色为黑色
        tickcolor='black',  # 设置y轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线       # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2
    )
)
pio.write_image(fig1, file_path+'_VAR.png')
# 显示保存成功的消息
print(f'Figure2 saved to {file_path}')
#fig1.show()



# 创建一个图形对象
fig2 = go.Figure()

if "ac1" in state_df.columns:
    fig2.add_trace(
        go.Scatter(
            x=state_df.time.values,
            y=state_df["ac1"].values,
            name="autocorrelation",
            line_color='firebrick',  # 指定轨迹的颜色
            line_width=3.5 # 设置轨迹的宽度为3
        )
    )

# 更新图形的布局
fig2.update_layout(
    xaxis_range=[int(T1*len1), int(T2*len1)],  # 设置x轴范围为完整数据集的范围
    showlegend=True,  # 设置为False以隐藏图例    legend_traceorder='normal',
    plot_bgcolor='white',  # 设置图形背景颜色为白色
    xaxis=dict(
        gridcolor='white',  # 设置x轴背景颜色为白色
        color='black',  # 设置x轴刻度和标签颜色为黑色
        linecolor='black',  # 设置x轴线条颜色为黑色
        tickcolor='black',  # 设置x轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线        # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2

    ),
    yaxis=dict(
        gridcolor='white',  # 设置y轴背景颜色为白色
        color='black',  # 设置y轴刻度和标签颜色为黑色
        linecolor='black',  # 设置y轴线条颜色为黑色
        tickcolor='black',  # 设置y轴刻度线颜色为黑色
        showgrid=True,
        showticklabels=True,  # 隐藏y轴的数据标签 # 不显示网格线       # 不显示网格线
        linewidth=3.5      # 设置x轴线条宽度为2

    )
)
#fig2.show()

pio.write_image(fig2, file_path+'_AC1.png')
# 显示保存成功的消息
print(f'Figure2 saved to {file_path}')

