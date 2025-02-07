#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make fig with rows
- simulation overlayed with bifurcation diagram
- dl preds
这段代码的作用是生成一个复杂的图表，该图表展示了在模拟数据上应用深度学习模型预测（DL predictions）的结果，
并将其与分叉图（bifurcation diagrams）进行叠加比较。图表的布局分为三行三列，
分别对应不同的模拟情况：鞍点-节点分叉（saddle-node bifurcation）、叉点分叉（pitchfork bifurcation）和空模型（null）。

具体步骤如下：
1. **导入库**：代码开始部分导入了所需的Python库，包括NumPy、Pandas、Plotly等。
2. **加载数据**：从CSV文件中加载模拟数据和分叉数据。
3. **处理数据**：对数据进行预处理，包括计算DL概率、调整时间轴等。
4. **创建颜色方案**：定义了用于图表中不同元素的颜色。
5. **创建子图布局**：使用Plotly的`make_subplots`函数创建3x3的子图布局。
6. **绘制分叉图**：为每种分叉类型绘制分叉图，展示系统状态的变化。
7. **添加模拟轨迹和平滑曲线**：为每种情况添加模拟的细胞状态轨迹和平滑曲线。
8. **添加DL预测结果**：将深度学习模型的预测结果添加到图表中，包括不同类型分叉的概率。
9. **设置图表样式**：定义字体大小、线条宽度、标记大小等样式设置。
10. **添加注释和标题**：为图表的每个部分添加注释、标题和箭头。
11. **设置轴属性**：配置每个子图的轴属性，包括标题、范围、刻度等。
12. **导出图表**：将图表导出为PNG和PDF格式的文件。
这段代码对应于论文中的“Results”或“Figures”部分，具体可能是用于展示论文中提到的模型预测细胞状态转变的能力。
图表可能用于比较模型预测与实际观测数据之间的一致性，以及展示不同类型细胞状态转变的动态过程。
通过这种方式，研究者可以直观地展示他们的模型如何有效地识别和预测细胞状态的转变点。
Columns:
- saddle-node bifurcation
- pitchfork bifurcation

"""


import time

start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Load in data
df_fold = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_fold_forced.csv")
df_pf = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_pf_forced.csv")
df_null = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_null.csv")

df_fold_bif = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_fold_bif.csv")
df_pf_bif = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_pf_bif.csv")
df_null_bif = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/df_null_bif.csv")


# DL probability for *any* bifurcation
bif_labels = ["1", "2", "3",'4','5']
df_fold["any"] = df_fold[bif_labels].dropna().sum(axis=1)
df_pf["any"] = df_pf[bif_labels].dropna().sum(axis=1)
df_null["any"] = df_null[bif_labels].dropna().sum(axis=1)


# Time column for bif data
m1start = 1
m1end = 4.75
m = 750 / (m1end - m1start)

df_fold_bif["time"] = df_fold_bif["m1"].apply(lambda x: m * (x - m1start))
df_null_bif["time"] = df_null_bif["m1"].apply(lambda x: m * (x - m1start))


kstart = 1
kend = 1 / 4
m = 750 / (kend - kstart)
df_pf_bif["time"] = df_pf_bif["k"].apply(lambda x: m * (x - kstart))


# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray


col_other_bif = "gray"
dic_colours = {
    "state": "rosybrown",
    "smoothing": "red",
    "variance": cols[1],
    "ac": cols[2],
    "dl_any": cols[0],
    "dl_fold": cols[4],
    "dl_other": col_other_bif,
    "dl_ns": col_other_bif,
    "dl_fold": cols[1],
    "dl_tc": cols[3],
    "dl_pf": cols[2],
    "dl_null": cols[0],
    "bif": "black",
}


fig_height = 550
fig_width = 800

font_size = 12
font_family = "Times New Roman"
font_size_letter_label = 12
font_size_titles = 16

linewidth = 3
linewidth_axes = 3.5
tickwidth = 3.5
ticklen = 5

marker_size = 2.0

# Opacity of DL probabilities for different bifs
opacity = 1

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600

fig = make_subplots(
    rows=3,
    cols=3,
    shared_xaxes=True,
    vertical_spacing=0.04,
)


# ----------------
# Col 1: fold
# ------------------

col = 1


# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="0_real"')["time"],
        y=df_fold_bif.query('variable=="0_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="3_real"')["time"],
        y=df_fold_bif.query('variable=="3_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth, "dash": "dash"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="4_real"')["time"],
        y=df_fold_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="2_real"')["time"],
        y=df_fold_bif.query('variable=="2_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)


df = df_fold
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        mode="markers",
        marker=dict(size=marker_size),
        showlegend=False,
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        name="Null",
        showlegend=True,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# # Weight for any bif
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['any'],
#                marker_color=dic_colours['dl_any'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# # Weight for PD
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['1'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )

# # Weight for NS
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['2'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                opacity=opacity,
#                ),
#     row=3,col=col,
#     )

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        # showlegend=False,
        name="Fold",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
        name="TC",
    ),
    row=3,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# ----------------
# Col 2: pf
# ------------------

col = 2


# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="0_real"')["time"],
        y=df_pf_bif.query('variable=="0_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="2_real" and time>500')["time"],
        y=df_pf_bif.query('variable=="2_real" and time>500')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="3_real"')["time"],
        y=df_pf_bif.query('variable=="3_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth, "dash": "dash"},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="4_real"')["time"],
        y=df_pf_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


df = df_pf
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        mode="markers",
        marker=dict(size=marker_size),
        showlegend=False,
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# # Trace for variance
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['variance'],
#                marker_color=dic_colours['variance'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=2,col=col,

#     )


# # Trace for lag-1 AC
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['ac1'],
#                marker_color=dic_colours['ac'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,

#     )


# # Weight for any bif
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['any'],
#                marker_color=dic_colours['dl_any'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# # Weight for PD
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['1'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )

# # Weight for NS
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['2'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                opacity=opacity,
#                ),
#     row=3,col=col,
# )

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)

# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        # showlegend=False,
        name="PF",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=True,
        name="TC",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# ----------------
# Col 3: null
# ------------------

col = 3


# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_null_bif.query('variable=="4_real"')["time"],
        y=df_null_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


df = df_null
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        mode="markers",
        marker=dict(size=marker_size),
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# # Weight for any bif
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['any'],
#                marker_color=dic_colours['dl_any'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# # Weight for PD
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['1'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )

# # Weight for NS
# fig.add_trace(
#     go.Scatter(x=df['time'],
#                y=df['2'],
#                marker_color=dic_colours['dl_other'],
#                showlegend=False,
#                line={'width':linewidth},
#                opacity=opacity,
#                ),
#     row=3,col=col,
#     )

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        name="Fold",
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        # showlegend=False,
        # name='Other',
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        # name='Pitchfork',
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# #--------------
# # Shapes
# #--------------

# list_shapes = []


# # Vertical lines for where transitions occur
# t_transition = 500

# #  Line for PD transition
# shape = {'type': 'line',
#           'x0': t_transition,
#           'y0': 0,
#           'x1': t_transition,
#           'y1': 1,
#           'xref': 'x',
#           'yref': 'paper',
#           'line': {'width':linewidth,'dash':'dot'},
#           }
# list_shapes.append(shape)

# #  Line for NS transition
# shape = {'type': 'line',
#           'x0': t_transition,
#           'y0': 0,
#           'x1': t_transition,
#           'y1': 1,
#           'xref': 'x5',
#           'yref': 'paper',
#           'line': {'width':linewidth,'dash':'dot'},
#           }
# list_shapes.append(shape)


# #  Line for Fold transition
# shape = {'type': 'line',
#           'x0': t_transition,
#           'y0': 0,
#           'x1': t_transition,
#           'y1': 1,
#           'xref': 'x9',
#           'yref': 'paper',
#           'line': {'width':linewidth,'dash':'dot'},
#           }
# list_shapes.append(shape)


# #  Line for TC transition
# shape = {'type': 'line',
#           'x0': t_transition,
#           'y0': 0,
#           'x1': t_transition,
#           'y1': 1,
#           'xref': 'x13',
#           'yref': 'paper',
#           'line': {'width':linewidth,'dash':'dot'},
#           }
# list_shapes.append(shape)


# #  Line for PF transition
# shape = {'type': 'line',
#           'x0': t_transition,
#           'y0': 0,
#           'x1': t_transition,
#           'y1': 1,
#           'xref': 'x17',
#           'yref': 'paper',
#           'line': {'width':linewidth,'dash':'dot'},
#           }
# list_shapes.append(shape)


# fig['layout'].update(shapes=list_shapes)


# --------------
# Add annotations
# ----------------------

list_annotations = []


# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 10)]
axes_numbers[0] = ""
idx = 0
for axis_number in axes_numbers:
    label_annotation = dict(
        x=0.01,
        y=1.00,
        text="({})".format(label_letters[idx]),
        xref="x{} domain".format(axis_number),
        yref="y{} domain".format(axis_number),
        showarrow=False,
        font=dict(color="black", size=font_size_letter_label),
    )
    list_annotations.append(label_annotation)
    idx += 1


# Bifurcation titles
y_pos = 1.06
title_fold = dict(
    x=0.5,
    y=y_pos,
    text="Fold",
    xref="x domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

title_pf = dict(
    x=0.5,
    y=y_pos,
    text="Pitchfork",
    xref="x2 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

title_null = dict(
    x=0.5,
    y=y_pos,
    text="Null",
    xref="x3 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

# # label for scaling factor of variance (10^-3)
# axes_numbers = ['7','8','9']
# for axis_number in axes_numbers:
#     label_scaling = dict(
#         x=0,
#         y=1,
#         text='&times;10<sup>-3</sup>',
#         xref='x{} domain'.format(axis_number),
#         yref='y{} domain'.format(axis_number),
#         showarrow=False,
#         font = dict(
#                 color = "black",
#                 size = font_size)
#         )
#     list_annotations.append(label_scaling)


# Arrows to indiciate rolling window
axes_numbers = [7, 8, 9]
arrowhead = 1
arrowsize = 2
arrowwidth = 0.5

for axis_number in axes_numbers:
    # Make right-pointing arrow
    annotation_arrow_right = dict(
        x=0,  # arrows' head
        y=0.1,  # arrows' head
        ax=100,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )
    # Make left-pointing arrow
    annotation_arrow_left = dict(
        ax=0,  # arrows' head
        y=0.1,  # arrows' head
        x=100,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )

    # Append to annotations
    list_annotations.append(annotation_arrow_left)
    list_annotations.append(annotation_arrow_right)


# list_annotations.append(label_annotation)

list_annotations.append(title_fold)
list_annotations.append(title_pf)
list_annotations.append(title_null)


fig["layout"].update(annotations=list_annotations)


# -------------
# Axes properties
# -----------

# Global y axis properties
fig.update_yaxes(
    showline=True,
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=yaxes_standoff,
)

# Global x axis properties
fig.update_xaxes(
    range=[0, 750],
    showline=True,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=xaxes_standoff,
)


# Specific x axes properties
fig.update_xaxes(
    title="pseudotime",
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    row=3,
)

# Specific y axes properties
fig.update_yaxes(title="Gene 1", row=1, col=1)
fig.update_yaxes(title="Gene 1", row=2, col=1)
fig.update_yaxes(title="DL probability", row=3, col=1)


fig.update_yaxes(range=[0, 4.5], row=1, col=1)
fig.update_yaxes(range=[0, 4.5], row=2, col=1)

fig.update_yaxes(range=[0, 2], row=1, col=2)
fig.update_yaxes(range=[0, 2], row=2, col=2)


fig.update_yaxes(range=[0, 1], row=1, col=3)
fig.update_yaxes(range=[0, 1], row=2, col=3)


fig.update_yaxes(range=[-0.05, 1.05], row=3)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 50, "r": 5, "b": 50, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)


fig.update_layout(
    legend=dict(
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=1,
    )
)


fig.update_traces(connectgaps=True)

# Export as temp image
fig.write_html('/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/temp.html')
fig.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/freedman_model_preds.png", scale=2)
fig.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/freedman_model_preds.pdf", scale=2)


# # Import image with Pil to assert dpi and export - this assigns correct
# # dimensions in mm for figure.
# from PIL import Image
# img = Image.open('/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/temp.png')
# dpi=96*8 # (default dpi) * (scaling factor)
# img.save('../../results/figure_2.png', dpi=(dpi,dpi))


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))
