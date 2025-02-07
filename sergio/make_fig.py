
import time

start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# -----------------
# Read in trajectory data
# -----------------
df_traj = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/df_pca.csv")

# Colour scheme
cols = px.colors.qualitative.Plotly  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

col_other_bif = "gray"
dic_colours = {
    "state": "gray",
    "smoothing": "black",
    "variance": cols[1],
    "ac": cols[2],
    "dl_any": cols[0],
    "dl_pd": col_other_bif,
    "dl_ns": cols[1],
    "dl_fold": cols[1],
    "dl_trans": cols[3],
    "dl_pf": cols[2],
    "dl_null": cols[0],
    "bif": "black",
}

fig_height = 400
fig_width = 700

font_size = 16
font_family = "Times New Roman"
font_size_letter_label = 16
font_size_titles = 18

linewidth = 1
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 5
marker_size = 2.5

# Opacity of DL probabilities for different bifs
opacity = 1

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0

# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


# plot as a function of pseudotime or pseudotime index
xvar = "pseudotime"


# Transition time is taken as first time where class 6 appears
transition = df_traj.query("pca_comp==0 and cluster==6")["pseudotime_index"].iloc[0]
transition_pseudo = df_traj.query("pca_comp==0 and cluster==6")["pseudotime"].iloc[0]

fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.04,
)


# ----------------
# Col 1: forced
# ----------------

col = 1

# Trajectory of class 5 cells
df_plot = df_traj.query(
    "cluster==5 and pca_comp==0 and pseudotime_index<=@transition"
).copy()
# Reset pseudotime index to start at zero
first_idx = df_plot["pseudotime_index"].iloc[0]
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color=dic_colours["state"],
        showlegend=True,
        legend="legend",
        name="class 5",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)


# Trajectory of class 6 cells
df_plot = df_traj.query("cluster==6 and pca_comp==0").copy()
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color='peru',
        showlegend=True,
        name="class 6",
        legend="legend",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# Trajectory of class 3 cells
df_plot = df_traj.query("cluster==3 and pca_comp==0").copy()
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color='brown',
        showlegend=True,
        legend="legend",
        name="class 3",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# Trajectory of class 4 cells
df_plot = df_traj.query("cluster==4 and pca_comp==0").copy()
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color='darkmagenta',
        showlegend=True,
        legend="legend",
        name="class 4",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)


# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 3)]
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
    fig["layout"].update(annotations=[label_annotation])
    idx += 1


# Vertical lines for where transitions occur
list_shapes = []
shape = {
    "type": "line",
    "x0": transition_pseudo if xvar == "pseudotime" else transition,
    "y0": 0,
    "x1": transition_pseudo if xvar == "pseudotime" else transition,
    "y1": 1,
    "xref": "x",
    "yref": "paper",
    "line": {"width": 1, "dash": "dash"},
}
list_shapes.append(shape)
fig["layout"].update(shapes=list_shapes)


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
    title="Pseudotime",
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    row=1,
    range=[0, 450] if xvar == "pseudotime_index" else [0.5, 0.78],
)

# Specific y axes properties
fig.update_yaxes(title="First PCA", row=1, col=1)
fig.update_yaxes(range=[-7.5, 2.4], row=1)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 60, "r": 5, "b": 60, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)


fig.update_layout(
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.75,
        xanchor="right",
        x=1,
        font=dict(size=14),
        itemsizing="constant",
    )
)

fig.show()
# Export as temp image
fig.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/fig_sergio_first_row.png", scale=2)



# ----------------
# Col 2: null
# ----------------
df_ews_null = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/ews/df_ews_null.csv")
df_ews_null=df_ews_null.iloc[:150,:]
# Colour scheme
cols = px.colors.qualitative.Plotly  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

col_other_bif = "gray"


fig_height = 400
fig_width = 700

font_size = 16
font_family = "Times New Roman"
font_size_letter_label = 16
font_size_titles = 18

linewidth = 1
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 5
marker_size = 2.5

# Opacity of DL probabilities for different bifs
opacity = 1

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0

# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


# plot as a function of pseudotime or pseudotime index
xvar = "pseudotime"


fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.04,
)
col = 2

# Trajectory
fig.add_trace(
    go.Scatter(
        x=df_ews_null[xvar],
        y=df_ews_null["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)


# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 3)]
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
    fig["layout"].update(annotations=[label_annotation])
    idx += 1


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
    title="Pseudotime",
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    row=1,
    col=2,
    range=[0, 450] if xvar == "pseudotime_index" else [0.5, 0.78],
)

# Specific y axes properties
fig.update_yaxes(title="State", row=1, col=2)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 60, "r": 5, "b": 60, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

fig.show()
# Export as temp image
fig.write_image("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/fig_sergio_first_col2.png", scale=2)