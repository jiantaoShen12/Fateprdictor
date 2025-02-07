#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:49:05 2023

Example application of deep learning classifier to single time series

"""


import numpy as np
import pandas as pd
import ewstools

from tensorflow.keras.models import load_model



for a in range(1,4):
# Load models
    if  a == 1:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_chick_heart/output/LSTM/"
    elif a == 2:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_chick_heart/output/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        c2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_chick_heart/output/BILSTM/"
   

    # Import data
    tsid = 14
    df = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/test_chick_heart/df_chick.csv").set_index("Beat number")
    series = df.query('tsid==@tsid and type=="pd"')["IBI (s)"]

    # Set up time series object
    ts = ewstools.TimeSeries(series, transition=300)

# Detrend
    ts.detrend(method="Gaussian", bandwidth=20)

    # Compute variance and lag-1 autocorrelation
    ts.compute_var(rolling_window=0.5)
    ts.compute_auto(rolling_window=0.5, lag=1)

    # Compute DL predictions
    ts.apply_classifier_inc(c1, inc=10, name="c1", verbose=0)
    ts.apply_classifier_inc(c2, inc=10, name="c2", verbose=0)
    df.dl=ts.dl_preds
    print(ts.dl_preds)
    df.dl.to_csv(path_out+f"df_dl_{tsid}.csv",index=False)
    fig = ts.make_plotly(ens_avg=True)
    fig.show()
    if  a == 1:
        df_ews_forced = ts.state
        df_ews_forced = df_ews_forced.join(ts.ews)
        df_ews_forced.to_csv(path_out+f"df_ews_{tsid}.csv")

# new_names = {
#     "state": "state",
#     "smoothing": "smoothing",
#     "variance": "variance",
#     "ac1": "ac1",
#     "DL class 0": "Null",
#     "DL class 1": "Period-doubling",
#     "DL class 2": "Neimark-Sacker",
#     "DL class 3": "Fold",
#     "DL class 4": "Transcritical",
#     "DL class 5": "Pitchfork",
# }
# fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))

# fig.write_html("output/example.html")
