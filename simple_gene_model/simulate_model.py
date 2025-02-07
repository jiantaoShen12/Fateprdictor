#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:29:12 2022

Simulate single trajectories of 2d gene model from Freedman et al.
https://journals.biologists.com/dev/article/150/11/dev201280/312613/A-dynamical-systems-treatment-of-transcriptomic

Compute EWS for g1 and g2
这段代码的作用是模拟一个二维基因模型的单个轨迹，并计算早期预警信号（EWS）以探测模型中的临界转变（例如分叉）。具体步骤如下：

1. **导入库**：代码开始部分导入了NumPy、Pandas、Matplotlib、ewstools、TensorFlow等库。

2. **设置日志记录级别**：设置TensorFlow的日志记录级别为ERROR，以避免显示警告信息。
3. **加载深度学习模型**：加载之前训练好的深度学习分类器模型。
4. **定义模拟函数**：`simulate_model`函数用于模拟基因模型的动态行为，生成时间序列数据。
5. **模拟鞍点-节点分叉（saddle-node bifurcation）轨迹**：设置参数模拟鞍点-节点分叉，包括分叉点、时间范围、噪声水平等。
6. **生成伪时间序列**：通过多个模拟轨迹的数据，生成伪时间序列，用于后续分析。
7. **计算EWS**：对特定基因（g1和g2）的时间序列计算EWS，使用`ewstools.TimeSeries`类及其方法进行去趋势、计算方差、自相关，并应用深度学习分类器。
8. **生成分叉图**：计算并生成分叉图，展示不同参数下的稳定和不稳定状态。
9. **导出数据**：将计算得到的EWS结果和分叉图数据导出为CSV文件。
这段代码对应于论文中关于使用动态系统理论来分析转录组轨迹的部分。具体来说，它可能对应于论文中的以下内容：
- **模拟动态系统**：代码模拟了细胞状态转变的动态过程，这与论文中提到的Waddington景观和细胞命运转变的概念相符。
- **计算EWS**：代码中的EWS计算与论文中提到的用于探测细胞状态转变的临界点的方法相对应。
- **分叉图**：生成的分叉图展示了模型预测的分叉行为，这与论文中讨论的细胞命运决策和多稳定性分析相一致。
总的来说，这段代码实现了论文中提到的动态系统理论在分析单细胞转录组数据中的应用，特别是在探测和理解细胞状态转变的临界点方面。通过模拟和EWS分析，研究者可以更好地理解细胞分化过程中的动态变化，并识别控制细胞命运的关键基因和分子机制。
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import python libraries
import numpy as np
import pandas as pd
import ewstools
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model
import plotly.express as px

h=1
for a in range(1,4):
# Load models
    if  a == 1:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/LSTM/"
    elif a == 2:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/simple_gene_model/output/BILSTM/"



    def simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=1):
        # Simulation parameters
        dt = 0.01
        t0 = 0
        tburn = 100

        # initialise DataFrame for each variable to store all realisations
        df_sims_x = pd.DataFrame([])
        df_sims_y = pd.DataFrame([])

        # Initialise arrays to store single time-series data
        t = np.arange(t0, tmax, dt)
        g1 = np.zeros(len(t))
        g2 = np.zeros(len(t))

        # Model parameters
        if type(m1) == list:
            m1 = np.linspace(m1[0], m1[1], len(t))
        else:
            m1 = np.ones(int(tmax / dt)) * m1

        if type(k) == list:
            k = np.linspace(k[0], k[1], len(t))
        else:
            k = np.ones(int(tmax / dt)) * k

        def de_fun_g1(g1, g2, m1, k):
            return m1 / (1 + g2**2) - k * g1

        def de_fun_g2(g1, g2, m2, k):
            return m2 / (1 + c * g1**2) - k * g2

        # Create brownian increments (s.d. sqrt(dt))
        dW_x_burn = np.random.normal(
            loc=0, scale=sigma_x * np.sqrt(dt), size=int(tburn / dt)
        )
        dW_x = np.random.normal(loc=0, scale=sigma_x * np.sqrt(dt), size=len(t))

        dW_y_burn = np.random.normal(
            loc=0, scale=sigma_y * np.sqrt(dt), size=int(tburn / dt)
        )
        dW_y = np.random.normal(loc=0, scale=sigma_y * np.sqrt(dt), size=len(t))

        # Run burn-in period on x0
        for i in range(int(tburn / dt)):
            g10 = g10 + de_fun_g1(g10, g20, m1[0], k[0]) * dt + dW_x_burn[i]
            g20 = g20 + de_fun_g2(g10, g20, m2, k[0]) * dt + dW_y_burn[i]

        # Initial condition post burn-in period
        g1[0] = g10
        g2[0] = g20

        # Run simulation
        for i in range(len(t) - 1):
            g1[i + 1] = g1[i] + de_fun_g1(g1[i], g2[i], m1[i], k[i]) * dt + dW_x[i]
            g2[i + 1] = g2[i] + de_fun_g2(g1[i], g2[i], m2, k[i]) * dt + dW_y[i]

        # Store series data in a temporary DataFrame
        data = {"time": t, "g1": g1, "g2": g2}
        df = pd.DataFrame(data)

        return df


    # -------------
    # Saddle node trajectory
    # --------------
    np.random.seed(7)

    ncells = 20
    tmax = 750
    g10 = 0
    g20 = 0
    m1bif = 3.5
    m1start = 1
    m1end = m1start + (m1bif - m1start) * 1.5

    m2 = 3
    k = 1
    c = 1
    sigma_x = 0.05
    sigma_y = 0.05


    list_df_traj = []

    for cell in range(ncells):
        # Forced trajectory
        m1 = [m1start, m1end]
        df = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax)
        df = df.iloc[::100]

        df["cell"] = cell
        list_df_traj.append(df)

        print("Simulation done for cell {}".format(cell))

    df_traj = pd.concat(list_df_traj)


    # Stitch together individual cell trajectories to make pseudotime trajectories
    mat_cell_g1 = (
        df_traj[["g1", "time", "cell"]].pivot(columns=["cell"], index="time").values
    )
    mat_cell_g2 = (
        df_traj[["g2", "time", "cell"]].pivot(columns=["cell"], index="time").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo_g1 = np.zeros((tmax, ncells))
    mat_pseudo_g2 = np.zeros((tmax, ncells))

    for j in range(ncells):
        for i in range(tmax):
            # take diagonal that loops around
            mat_pseudo_g1[i, j] = mat_cell_g1[i, (i + j) % ncells]
            mat_pseudo_g2[i, j] = mat_cell_g2[i, (i + j) % ncells]

    df_pseudo_g1 = pd.DataFrame(mat_pseudo_g1)
    df_pseudo_g1 = df_pseudo_g1.reset_index(names="time")
    df_pseudo_g1 = df_pseudo_g1.melt(id_vars="time", var_name="pseudo", value_name="g1")

    df_pseudo_g2 = pd.DataFrame(mat_pseudo_g2)
    df_pseudo_g2 = df_pseudo_g2.reset_index(names="time")
    df_pseudo_g2 = df_pseudo_g2.melt(id_vars="time", var_name="pseudo", value_name="g2")


    '''
    df_pseudo_g1 = df_pseudo_g1.iloc[h::2].reset_index(drop=True)
    df_pseudo_g2 = df_pseudo_g2.iloc[h::2].reset_index(drop=True)
    '''



    id_plot = 1
    # df_pseudo.query('pseudo==@id_plot')['x'].plot()

    ########## Compute EWS for g1
    series = df_pseudo_g1.query("pseudo==@id_plot")[["g1", "time"]].set_index("time")["g1"]
    ts = ewstools.TimeSeries(series,transition=500)
    ts.detrend(method='Gaussian', bandwidth=0.2)
    #ts.detrend(method="Lowess", span=0.2)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    ts.apply_classifier_inc(c1, inc=10, name="c1")
    ts.apply_classifier_inc(c2, inc=10, name="c2")

    # Make a temp figure to view
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    fig.show()
    fig.write_html(f'{path_out}ews_fold_g1.html')
    # Export ews
    df_dl = ts.dl_preds
    df_dl1=df_dl.copy()
    df_dl2=df_dl.copy()
    df_dl1=df_dl1[df_dl1['classifier'] == 'c1']
    df_dl2=df_dl2[df_dl2['classifier'] == 'c2']
    df_out = ts.state
    df_out = df_out.join(ts.ews, on="time")
    df_out1 = pd.merge(df_out,df_dl1,on=['time'],how='outer')
    df_out2 = pd.merge(df_out,df_dl2,on=['time'],how='outer')
    df_out = pd.concat([df_out1,df_out2],axis=0)#纵向合并
    df_out.to_csv(f'{path_out}df_fold_forced_g1.csv')

    ########## Compute EWS for g2
    series = df_pseudo_g2.query("pseudo==@id_plot")[["g2", "time"]].set_index("time")["g2"]
    ts = ewstools.TimeSeries(series,transition=500)
    ts.detrend(method='Gaussian', bandwidth=0.2)
    #ts.detrend(method="Lowess", span=0.2)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    ts.apply_classifier_inc(c1, inc=10, name="c1")
    ts.apply_classifier_inc(c2, inc=10, name="c2")

    # Make a temp figure to view
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    fig.show()
    fig.write_html(f'{path_out}ews_fold_g2.html')
    # Export ews
    # Export ews
    df_dl = ts.dl_preds
    df_dl1=df_dl.copy()
    df_dl2=df_dl.copy()
    df_dl1=df_dl1[df_dl1['classifier'] == 'c1']
    df_dl2=df_dl2[df_dl2['classifier'] == 'c2']
    df_out = ts.state
    df_out = df_out.join(ts.ews, on="time")
    df_out1 = pd.merge(df_out,df_dl1,on=['time'],how='outer')
    df_out2 = pd.merge(df_out,df_dl2,on=['time'],how='outer')
    df_out = pd.concat([df_out1,df_out2],axis=0)#纵向合并
    df_out.to_csv(f'{path_out}df_fold_forced_g2.csv')


    # -----------------
    # Get bifurcation diagram
    # -----------------


    # Get roots of cubic polynomial
    m1_vals = np.linspace(m1start, m1end, 1000)

    list_roots_g1 = []
    list_roots_g2 = []

    for m1 in m1_vals:
        a0 = -m2 * (k**2)
        a1 = (k**3) + k * (m1**2)
        a2 = -2 * c * m2 * (k**2)
        a3 = 2 * c * (k**3)
        a4 = -m2 * (c**2) * (k**2)
        a5 = (c**2) * k**3

        p = [a5, a4, a3, a2, a1, a0]
        roots_g2 = np.roots(p)
        roots_g1 = m1 / (k * (1 + roots_g2**2))
        list_roots_g1.append(roots_g1)
        list_roots_g2.append(roots_g2)

    ar1 = np.array(list_roots_g1)
    ar2 = np.array(list_roots_g2)


    df_roots_g1 = pd.DataFrame()
    df_roots_g2 = pd.DataFrame()
    df_roots_g1["m1"] = m1_vals
    df_roots_g2["m1"] = m1_vals

    for i in np.arange(5):
        df_roots_g1[str(i)] = ar1[:, i]
        df_roots_g2[str(i)] = ar2[:, i]


    def get_real(col):
        np.imag(col) == 0

        np.real(col)

        real_vals = np.real(col)

        pure_real = real_vals.copy()
        pure_real[np.imag(col) != 0] = np.nan

        out = pd.Series(pure_real, index=col.index)
        return out


    cols = df_roots_g1.columns
    for col in cols:
        df_roots_g1["{}_real".format(col)] = get_real(df_roots_g1[col])
        df_roots_g2["{}_real".format(col)] = get_real(df_roots_g2[col])

    df_plot_g1 = df_roots_g1.melt(
        id_vars="m1", value_vars=["0_real", "1_real", "2_real", "3_real", "4_real"]
    )
    df_plot_g2 = df_roots_g2.melt(
        id_vars="m1", value_vars=["0_real", "1_real", "2_real", "3_real", "4_real"]
    )

    # make fig
    fig = px.line(df_plot_g2, x="m1", y="value", color="variable")
    fig.write_html(f'{path_out}bif_fold.html')

    # Export
    df_plot_g1.dropna(inplace=True)
    df_plot_g1.to_csv(f'{path_out}df_fold_bif_g1.csv', index=False)
    df_plot_g2.dropna(inplace=True)
    df_plot_g2.to_csv(f'{path_out}df_fold_bif_g2.csv', index=False)
