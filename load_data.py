import pandas as pd
import csv
import numpy as np
import os

code_num = 758
trade_day = 1702
fts = 10   #feature size
def load_EOD_data():
    f = open(r'../data/result(758)_label.csv')
    df = pd.read_csv(f, header=None)
    data = df.iloc[:, 0:10].values  #1-10 columns all rows
    eod_data = data.reshape(code_num, trade_day, fts) #ndarray (758, 1702, 10)
    data_label = df.iloc[:, -1].values
    ground_truth = data_label.reshape(code_num, trade_day)
    # file = open('eod.csv', "a")
    # file.write(str(eod_data))
    # file.close
    # file = open('ground.csv', "a")
    # file.write(str(ground_truth))
    # file.close
    return eod_data, ground_truth

def get_batch(eod_data, gt_data, offset, seq_len):
    return eod_data[:, offset:offset + seq_len, :], \
           gt_data[:, offset + seq_len]

def get_fund_adj_H():
    tensor = np.random.rand(0, code_num, 62)   #(0,758,62)
    path = '../data/H_fund_adj1'
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('_')[0]))
    for filename in path_list:
        df = pd.read_csv(open(os.path.join(path, filename)), header=None)
        fund_adj = df.values
        tensor = np.concatenate((tensor, fund_adj[None]), axis=0)
    return tensor

def get_industry_adj():
    adj = np.mat(pd.read_csv(open(r'../data/industry_adj(758).csv'), header=None))
    return adj