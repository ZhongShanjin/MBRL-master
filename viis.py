import pickle
import os, sys
import pandas as pd
import torch
import json
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import requests
import seaborn as sns
import torch.nn as nn
import torch.optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering		#导入层次聚类模块

pk_dir=Path('exps/50/motif/predcls/lt/internal/relabel/em_E.pk_topk_0.7')
dict_file=Path('datasets/vg/50/VG-SGG-dicts-with-attri.json')
# # 读取数据集
dics = pickle.load(open(pk_dir, "rb"))
# dic = pickle.load(open("demo/custom_prediction.pk", "rb"))
# VG50标注
vocab = json.load(open("datasets/vg/50/VG-SGG-dicts-with-attri.json"))
thrdim_df = pd.read_csv('thrdim.csv')
x=thrdim_df.loc[:,['osubxc:objxc','subyc:objyc','subarea:objarea']]
top_k=10000
rel_split=5
y_pred = DBSCAN(eps = 0.11, min_samples = 10).fit_predict(x) # y_pred保存了类别值
labels=y_pred

plt.figure(figsize=(40,40))
ax1 = plt.axes(projection='3d')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
cm = plt.cm.get_cmap('jet')
ax=ax1.scatter(thrdim_df['osubxc:objxc'][:top_k],thrdim_df['subyc:objyc'][:top_k],thrdim_df['subarea:objarea'][:top_k],c=labels[:top_k],cmap=cm)
plt.colorbar(ax)
plt.show()