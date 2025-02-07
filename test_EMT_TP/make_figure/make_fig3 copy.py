#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make fig of ROC curves with inset showing histogram of highest DL probability
"""


import time

start_time = time.time()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

# Import PIL for image tools
from PIL import Image

# -----------
# General fig params
# ------------

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "dl_bif": cols[1],
    "dl_bif1": cols[3],
    "variance": cols[0],
    "ac": cols[2],
    "dl_fold": cols[3],
    "dl_hopf": cols[4],
    "dl_branch": cols[5],
    "dl_null": "black",
}

# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 3  # 3 panels wide
fig_height = fig_width


font_size = 10
font_family = "Times New Roman"
font_size_letter_label = 14
font_size_auc_text = 12


# AUC annotations
x_auc = 0.98
y_auc = 0.6
x_N = 0.18
y_N = 0.05
y_auc_sep = 0.1

linewidth = 1.7  # 原来的值是0.7，现在加倍
linewidth_axes = 1.5  # 原来的值是0.5，现在加倍
tickwidth = 1.5   # 原来的值是0.5，现在加倍
linewidth_axes_inset = 1.5   # 原来的值是0.5，现在加倍

axes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


def make_roc_figure(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif"]
    auc_dl = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
            width=linewidth * 1.5,  # 增加线宽，例如将原来的linewidth乘以2
                color=dic_colours["dl_bif"],
            ),
        )
    )

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif1"]
    auc_dl1 = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["dl_bif1"],
            ),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["variance"],
            ),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["ac"],
            ),
        )
    )

    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line=dict(
                color="black",
                dash="dot",
                width=linewidth,
            ),
        )
    )

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    # label_annotation = dict(
    #     # x=sum(xrange)/2,
    #     x=0.02,
    #     y=1,
    #     text="<b>{}</b>".format(letter_label),
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     font=dict(
    #         color='black',
    #         size=font_size_letter_label,
    #     ),
    # )

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc-y_auc_sep*2,
        text="A<sub>Fatepredictor</sub>={:.2f}".format(auc_dl),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_dl1 = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*3.3,
        text="A<sub>Deep Learning</sub>={:.2f}".format(auc_dl1),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif1"],
            size=font_size_auc_text,
        ),
    )


    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*4.2,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["variance"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ac = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc -  y_auc_sep*5,
        text="A<sub>AC</sub>={:.2f}".format(auc_ac),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["ac"],
            size=font_size_auc_text,
        ),
    )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )
    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    #list_annotations.append(label_annotation)
    list_annotations.append(annotation_auc_dl)
    list_annotations.append(annotation_auc_dl1)
    list_annotations.append(annotation_auc_var)
    list_annotations.append(annotation_auc_ac)
    list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    return fig



def make_roc_figure2(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif"]
    auc_dl = df_trace.round(2)["auc"].iloc[0]


    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif1"]
    auc_dl1 = df_trace.round(2)["auc"].iloc[0]


    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
  

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
   

    # Line y=x


    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    # label_annotation = dict(
    #     # x=sum(xrange)/2,
    #     x=0.02,
    #     y=1,
    #     text="<b>{}</b>".format(letter_label),
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     font=dict(
    #         color='black',
    #         size=font_size_letter_label,
    #     ),
    # )

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc-y_auc_sep*2,
        text="A<sub>Fatepredictor</sub>={:.2f}".format(auc_dl),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_dl1 = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*3.3,
        text="A<sub>Deep Learning</sub>={:.2f}".format(auc_dl1),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif1"],
            size=font_size_auc_text,
        ),
    )


    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*4.2,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["variance"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ac = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc -  y_auc_sep*5,
        text="A<sub>AC</sub>={:.2f}".format(auc_ac),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["ac"],
            size=font_size_auc_text,
        ),
    )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )
    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    #list_annotations.append(label_annotation)
    list_annotations.append(annotation_auc_dl)
    list_annotations.append(annotation_auc_dl1)
    list_annotations.append(annotation_auc_var)
    list_annotations.append(annotation_auc_ac)
    list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    return fig


def make_roc_figure1(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif"]
    auc_dl = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
            width=linewidth * 1.5,  # 增加线宽，例如将原来的linewidth乘以2
                color=dic_colours["dl_bif"],
            ),
        )
    )

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif1"]
    auc_dl1 = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["dl_bif1"],
            ),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["variance"],
            ),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["ac"],
            ),
        )
    )

    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line=dict(
                color="black",
                dash="dot",
                width=linewidth,
            ),
        )
    )

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    # label_annotation = dict(
    #     # x=sum(xrange)/2,
    #     x=0.02,
    #     y=1,
    #     text="<b>{}</b>".format(letter_label),
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     font=dict(
    #         color='black',
    #         size=font_size_letter_label,
    #     ),
    # )

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc-y_auc_sep*2,
        text="A<sub>Fatepredictor</sub>={:.2f}".format(auc_dl),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_dl1 = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*3.3,
        text="A<sub>Deep Learning</sub>={:.2f}".format(auc_dl1),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["dl_bif1"],
            size=font_size_auc_text,
        ),
    )


    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep*4.2,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["variance"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ac = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc -  y_auc_sep*5,
        text="A<sub>AC</sub>={:.2f}".format(auc_ac),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["ac"],
            size=font_size_auc_text,
        ),
    )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )
    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    #list_annotations.append(label_annotation)
    #list_annotations.append(annotation_auc_dl)
    #list_annotations.append(annotation_auc_dl1)
    #list_annotations.append(annotation_auc_var)
    # list_annotations.append(annotation_auc_ac)
    #list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    return fig

import seaborn as sns





# -------
# Heart data
# --------
df_roc = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/df_roc.csv")
df_roc1 = pd.read_csv("sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/对比1/df_roc.csv")
# 将ews列中值为'DL bif'的所有行改为'DL bif1'
df_roc1=df_roc1.loc[df_roc1['ews'] == 'DL bif']
df_roc1["ews"]='DL bif1'
print(df_roc1)

df_combined = pd.concat([df_roc1, df_roc], ignore_index=True)
df_dl_forced = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/df_dl_fixed.csv")
print(df_combined)
df_combined.to_csv("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/make_figure/df_roc_combined.csv", index=False)

fig_roc = make_roc_figure(df_combined, "f", text_N="N={}".format(len(df_dl_forced) * 2))
fig_roc1 = make_roc_figure1(df_combined, "f", text_N="N={}".format(len(df_dl_forced) * 2))
fig_roc2 = make_roc_figure2(df_combined, "f", text_N="N={}".format(len(df_dl_forced) * 2))

fig_roc2.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/make_figure/temp_roc.png", scale=scale)
fig_roc1.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/make_figure/temp_roc1.png", scale=scale)
fig_roc2.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/make_figure/temp_roc2.png", scale=scale)



