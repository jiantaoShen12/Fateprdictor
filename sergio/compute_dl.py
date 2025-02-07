#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import tensorflow as tf
import ewstools
tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model

np.random.seed(0)

# Import PCA data
df = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/df_pca_3456.csv")
for a in range(1,4):
# Load models /home/sjt/workspace/汉字github/dl_train/
    if  a == 1:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/LSTM/"
    elif a == 2:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/BILSTM/"
    elif a == 4:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/RNN/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/RNN/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/RNN/"
    elif a == 0:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/"
    pca_comp = 0

    df_select = df.query("pca_comp==@pca_comp").copy()
    # 假设df_select是一个Pandas DataFrame
# 选择第3的倍数的行，例如第3行、第6行、第9行等

# 创建一个布尔索引，其中索引是3的倍数的位置为True
    is_multiple_of_three = (np.arange(len(df_select)) % 2 == 1)

# 使用布尔索引选择行
    df_select = df_select.iloc[is_multiple_of_three]
    series = df_select.reset_index()["value"]

    span = 0.2  # span of Lowess Filter
    inc=2
    # Compute EWS
    ts = ewstools.TimeSeries(series,transition=113)
    ts.detrend(method="Lowess", span=span)
    # ts.detrend(method='Gaussian', bandwidth=0.1)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    ts.apply_classifier_inc(c1, inc=inc, name="c1", verbose=0)
    ts.apply_classifier_inc(c2, inc=inc, name="c2", verbose=0)
    df_dl = ts.dl_preds
    # Make fig of EWS
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    fig.update_layout(title="PCA comp={}".format(pca_comp))
    fig.write_html(path_out+"temp.html")


    # Export
    df_dl["pseudotime"] = df_select["pseudotime"].iloc[df_dl["time"].values].values
    df_dl.to_csv(path_out+"df_dl.csv", index=False)
    
    df_ews = ts.state
    df_ews = df_ews.join(ts.ews).reset_index()  
    df_ews["pseudotime_index"] = df_ews["time"]
    df_ews["pseudotime"] = df_select["pseudotime"].iloc[df_ews["time"].values].values
    df_ews.to_csv("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/ews/df_ews.csv", index=False)
    # ----------------
    # Compute EWS for null time series - sample randomly from first 20% of residuals
    # and add to trend
    # -----------------

    # series = df_select.reset_index()['value_ro']
    # series = df_select.reset_index()['value_ro']
    series = df_select.reset_index()["value"]

    ts = ewstools.TimeSeries(series)
    ts.detrend(method="Lowess", span=span)
    resids = ts.state["residuals"]
    resids_sample = resids.iloc[: int(len(series) * 0.2)].sample(
        n=len(series), replace=True, ignore_index=True
    )
    series_null = ts.state["smoothing"] + resids_sample
    series_null.to_csv(path_out+"df_null.csv")

    # Compute EWS
    ts = ewstools.TimeSeries(series_null,transition=113)
    ts.detrend(method="Lowess", span=span)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)
    ts.apply_classifier_inc(c1, inc=inc, name="c1", verbose=0)
    ts.apply_classifier_inc(c2, inc=inc, name="c2", verbose=0)

    # Make fig of EWS
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    fig.write_html(path_out+"temp_null.html")

    # Add pseudotime col
    df_dl_null = ts.dl_preds
    df_dl_null["pseudotime_index"] = df_dl_null["time"]
    df_dl_null["pseudotime"] = (
        df_select["pseudotime"].iloc[df_dl_null["time"].values].values
    )

    df_ews_null = ts.state
    df_ews_null = df_ews_null.join(ts.ews).reset_index()
    df_ews_null["pseudotime_index"] = df_ews_null["time"]
    df_ews_null["pseudotime"] = (
        df_select["pseudotime"].iloc[df_ews_null["time"].values].values
    )

    # Export
    df_ews_null.to_csv("/home/sjt/workspace/beginning_project/cell_change尝试/sergio/output/ews/df_ews_null.csv", index=False)
    df_dl_null.to_csv(path_out+"df_null.csv", index=False)
