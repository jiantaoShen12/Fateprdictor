
import numpy as np
import pandas as pd
import ewstools
import tensorflow as tf
import os

tf.compat.v1.logging.set_verbosity(  tf.compat.v1.logging.ERROR)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model
#参数选择

# def zscore(s, window, thresh=2, return_all=False):
#     roll = s.rolling(window=window, min_periods=1, center=True)
#     avg = roll.mean()
#     std = roll.std(ddof=0)
#     z = s.sub(avg).div(std)
#     m = z.between(-thresh, thresh)

#     if return_all:
#     return z, avg, std, m
#     return s.where(m, avg)

np.random.seed(6)

# Import PCA data
df = pd.read_csv("/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/DE/trajectory_and_pseudotime.csv")
# 检查DataFrame是否为空
if df.empty:
    print("DataFrame is empty. Please check the file path and content.")
else:
    print("DataFrame is not empty and contains", len(df), "rows.")
inc =5
span=0.1
start_index =0
df_select=df
T=4
transition=260
#92 194 260 432 571

start_idx = max(transition - 499, 0)
df_select = df_select.iloc[start_idx:]
s=df_select["Feature1"].copy()
print(df_select)

for a in range(1,4):
    if  a == 1:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/LSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/LSTM/classifier_2.pkl"
        c2 = load_model(path)
        output_dir="/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/DE/LSTM/"
    elif a == 2:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/GRU/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/GRU/classifier_2.pkl"
        c2 = load_model(path)
        output_dir="/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/DE/GRU/"
    elif a == 3:
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_1.pkl"
        c1 = load_model(path)
        path = "/home/sjt/workspace/beginning_project//下载1/output_6(1.8)假20次pf_fold交错/BILSTM/classifier_2.pkl"
        c2 = load_model(path)
        output_dir="/home/sjt/workspace/beginning_project/cell_change尝试/test_DE/output/DE/BILSTM/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Compute EWS
    #print("Compute EWS for PCA comp {}".format(pca_comp))
    # time series increment for DL classifier
    #ts = ewstools.TimeSeries(s,transition=transition)
    ts = ewstools.TimeSeries(s,transition=transition)
    ts.detrend(method="Lowess", span=span)
    # ts.detrend(method='Gaussian', bandwidth=span)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)

    ts.apply_classifier_inc(c1, inc=inc, name="c1", verbose=0)
    ts.apply_classifier_inc(c2, inc=inc, name="c2", verbose=0)

    # Make quick fig of EWS
    fig = ts.make_plotly(ens_avg=True)
    fig.update_layout(height=650)
    #fig.update_layout(title="PCA comp={}".format(pca_comp))
    fig.write_html(f"{output_dir}temp.html")

    # Export data
    df_dl_forced = ts.dl_preds
    df_ews_forced = ts.state
    df_ews_forced = df_ews_forced.join(ts.ews)
    print(df_dl_forced.columns)
    #merged_df = pd.merge(df_dl_forced, df_select[['pseudotime1','pseudotime']], on='time', how='inner')
    df_dl_forced.to_csv(f"{output_dir}df_dl.csv", index=False)
    #merged_df = pd.merge(df_ews_forced, df_select[['pseudotime1','pseudotime']], on='time', how='inner')
    df_ews_forced.to_csv(f"{output_dir}df_ews.csv")
