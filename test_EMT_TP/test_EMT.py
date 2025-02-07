#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute DL and kendall tau at fixed evaluation points in chick heart data

"""


import time
start_time1 = time.time()

import sys

import numpy as np
import pandas as pd

import ewstools

import tensorflow as tf
from tensorflow.keras.models import load_model

import os

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_sims', type=int, help='Total number of model simulations', 
                    default=2500)
parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)

args = parser.parse_args()
model_sims = args.model_sims
use_inter_classifier = True if args.use_inter_classifier=='true' else False

np.random.seed(0)

eval_pts = np.arange(0.95, 1.05, 0.001) #  percentage of way through pre-transition time series
len_bi=[1,2,3,4]
len_null=[1.5,2.5,3.5,4.5]
transition=129
for a in range(1,5):
# Load models
    if  a == 1:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        m1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        m2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/LSTM/"
        bw=0.05
    elif a == 2:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        m1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        m2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        m1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        m2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/BILSTM/"
    elif a == 4:
        path = "/home/sjt/workspace/汉字github/dl_train/离散分岔的分类器/classifier_1.pkl"
        m1 = load_model(path)
        path = "/home/sjt/workspace/汉字github/dl_train/离散分岔的分类器/classifier_2.pkl"
        m2 = load_model(path)
        path_out="/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/对比/"  
        bw=0.2

    print('TF models loaded')

    list_ktau = []
    list_dl_preds = []
    for bi_tsid in len_bi:
    # Load in trajectory data
        df = pd.read_csv('/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/EMT_combined_with_pca_comp1.csv')
        df = df.assign(type="bifurcation")  # 给新列赋值为NaN
        # Load in transition times
        #--------------
        # period-doubling trajectories
        #---------------
        transition_sp=transition*bi_tsid+5
        start_time = max(transition_sp-500, 0)
        series = df.iloc[start_time:transition_sp]
        series=series["Feature1"].copy()
        transition1=min(transition_sp,500)
        # Compute EWS
        ts = ewstools.TimeSeries(series, transition=transition1)
        ts.detrend(method='Gaussian', bandwidth=bw)
        ts.compute_var(rolling_window=0.25)
        ts.compute_auto(rolling_window=0.25, lag=1)

        for eval_pt in eval_pts:
            eval_time = transition1*eval_pt
            if a==1:
                # Compute kendall tau at evaluation points
                ts.compute_ktau(tmin=0, tmax=eval_time)
                dic_ktau = ts.ktau
                dic_ktau['eval_time'] = eval_pt
                dic_ktau['tsid'] = bi_tsid
                list_ktau.append(dic_ktau)
            
            # Get DL predictions at eval pts
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)

            df_dl_preds = ts.dl_preds# use mean DL pred            
            df_dl_preds['eval_time']=eval_pt
            df_dl_preds['tsid'] = bi_tsid
            list_dl_preds.append(df_dl_preds)
            ts.clear_dl_preds()
    if a==1:
        df_ktau_forced = pd.DataFrame(list_ktau)
    df_dl_forced = pd.concat(list_dl_preds)

    list_ktau = []
    list_dl_preds = []
    for null_tsid in len_null:
    # Load in trajectory data
        df_null = pd.read_csv('/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/EMT_combined_with_pca_comp1.csv')
        df_null = df_null.assign(type="netural")  # 给新列赋值为NaN

        # Load in transition times
        #--------------
        # period-doubling trajectories

        transition_sp_null=transition*null_tsid+5
        start_time =int(max(transition_sp_null-500, 0)) 
        series = df.iloc[start_time:int(transition_sp_null)]
        series=series["Feature1"].copy()
        transition1=min(transition_sp_null,500)


        ts = ewstools.TimeSeries(series, transition=transition1)
        ts.detrend(method='Gaussian', bandwidth=bw)
        ts.compute_var(rolling_window=0.25)
        ts.compute_auto(rolling_window=0.25, lag=1)

        for eval_pt in eval_pts:
            eval_time = transition1*eval_pt
            if a==1:
                ts.compute_ktau(tmin=0, tmax=eval_time)
                dic_ktau = ts.ktau
                dic_ktau['eval_time'] = eval_pt
                dic_ktau['tsid'] = null_tsid
                list_ktau.append(dic_ktau)
            
            # Get DL predictions at eval pts
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name='m1', verbose=0)
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name='m2', verbose=0)

            df_dl_preds = ts.dl_preds# use mean DL pred            
            df_dl_preds['eval_time']=eval_pt
            df_dl_preds['tsid'] = null_tsid
            list_dl_preds.append(df_dl_preds)
            ts.clear_dl_preds()
    if a==1:
        df_ktau_null = pd.DataFrame(list_ktau)
    df_dl_null = pd.concat(list_dl_preds)

    # Export data
    if a==1:
        df_ktau_forced.to_csv('/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/df_ktau_pd_fixed.csv', index=False)
        df_ktau_null.to_csv('/home/sjt/workspace/beginning_project/cell_change尝试/test_EMT_TP/output/df_ktau_null_fixed.csv', index=False)
    df_dl_forced.to_csv(path_out+'df_dl_fixed.csv', index=False)
    df_dl_null.to_csv(path_out+'df_dl_null_fixed.csv', index=False)


    # Time taken for script to run
    end_time = time.time()
    time_taken = end_time - start_time1
    print('Ran in {:.2f}s'.format(time_taken))



