import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
#get_data
# def get_circle_dataset(n = 10):
#     dr,angle = np.random.randn(n)*0.05,1.5*np.pi*np.random.rand(n)
#     r = dr+1.3
#     x = r*np.sin(angle)
#     y = r*np.cos(angle)
#     data = np.concatenate([x[np.newaxis,:],y[np.newaxis,:]]).T
#     return data

# circle_data = get_circle_dataset()

my_data = pd.read_table("C:/Users/叶凯/Desktop/GARFIELD-NGS-master/datasets/ILM_INDEL_Training.txt")
# circle_data = np.array(circle_data)
# my_data.samples(5)
X_compare = my_data
y_compare = X_compare['Class']
x_compare = X_compare.iloc[:,1:11]
x = x_compare['DP']
y = x_compare['QD']
circle_data = np.concatenate([x[np.newaxis,:],y[np.newaxis,:]]).T
# print(data)
# x_train,x_test,y_train,y_test = train_test_split(x_compare,y_compare,test_size = 0.2,random_state = 5)

# plt.scatter(circle_data[:,0],circle_data[:,1])
# plt.show()

#feature selection


#preprocessing
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
transfer = MinMaxScaler(feature_range = (0,1))
# data = transfer.fit_transform(x_compare[['DP','QD']])
# print(data)
data = transfer.fit_transform(x_compare)
# print(data)
# plt.scatter(data)
# plt.show()
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import PowerTransformer
# from sklearn import preprocessing
# #标准化 去均值 方差规模化
#
# x_scale = preprocessing.scale(x)
# scaler = preprocessing.StandardScaler().fit(x)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# x_minmax = min_max_scaler.fit_transform(x)
# x_minmax
# # MaxAbsScaler
#
# # 数据会被规模化到[-1,1]之间(所有数据都会除以最大值，对那些已经中心化均值维0或者稀疏的数据有意义）
#
# # 规模化稀疏数据
# # 如果对稀疏数据进行去均值的中心化就会破坏稀疏的数据结构。虽然如此，我们也可以找到方法去对稀疏的输入数据进行转换，特别是那些特征之间的数据规模不一样的数据。
#
# # MaxAbsScaler 和 maxabs_scale这两个方法是专门为稀疏数据的规模化所设计的。
#
# # 规模化有异常值的数据
#
# # 在这里，你可以使用robust_scale 和 RobustScaler这两个方法。它会根据中位数或者四分位数去中心化数据。
#
# # 正则化Normalization
# x_normalized = preprocessing.normalize(x, norm='l2')
#
latent_size = 16

# Discriminator
D = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 10))

from torch.utils.data import DataLoader,TensorDataset

batch_size = 2
num_epochs = 2
real_data = torch.FloatTensor(data)
# print(real_data)
data_set = TensorDataset(real_data)
# print(data_set)
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
# print(data_loader)
loss_func = nn.MSELoss()
d_optimizer = torch.optim.Adam(D.parameters(),0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),0.0002)

# Start training
g_losses = []
d_losses = []

total_step = len(data_loader)
print(total_step)
for epoch in range(num_epochs):
    # print(epoch)
    for step, x in enumerate(data_loader):
        # print(x)
        x = x[0]
        # print(x)
        real_labels = torch.ones(batch_size, 1)
        print(real_labels)
        fake_labels = torch.zeros(batch_size, 1)
        # 为真实数据计算BCEloss
        outputs = D(x)
        d_loss_real = loss_func(outputs, real_labels)
        # 为假数据计算BCEloss
        z = torch.randn(batch_size, latent_size)
        fake_data = G(z)
        outputs = D(fake_data)
        d_loss_fake = loss_func(outputs, fake_labels)

        # 训练，只训练分类器
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        d_losses.append(d_loss.item())

        z = torch.randn(batch_size, latent_size)
        fake_data = G(z)
        outputs = D(fake_data)
        # 这里让生成器学习让损失函数朝着真样本的一侧移动
        g_loss = loss_func(outputs, real_labels)

        g_losses.append(g_loss.item())

        # 训练生成器
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


    if (epoch+1)%10==0:
        # 从生成器采样生成一些假数据点
        z = torch.randn(1000, latent_size)
        with torch.no_grad():
            fake_data = G(z)
        fake_x,fake_y = fake_data[:,0].numpy(),fake_data[:,1].numpy()
        real_x,real_y = real_data[:,0].numpy(),real_data[:,1].numpy()




        step = 0.02
        x = np.arange(-2,2,step)
        y = np.arange(-2,2,step)

        #将原始数据变成网格数据形式
        X,Y = np.meshgrid(x,y)
        n,m = X.shape
        #写入函数，z是大写
        inputs = torch.stack([torch.FloatTensor(X),torch.FloatTensor(Y)])
        inputs = inputs.permute(1,2,0)
        inputs = inputs.reshape(-1,2)
        with torch.no_grad():
            Z = D(inputs)
        Z = Z.numpy().reshape(n,m)

        plt.figure(figsize=(7,6))
        plt.title('Discriminator probablity')
        cset = plt.contourf(X,Y,Z,100)
        plt.colorbar(cset)
        # plt.show()

        plt.figure(figsize=(6,6))

        plt.scatter(real_x,real_y,c = 'w', edgecolor='b')
        plt.scatter(fake_x,fake_y,c = 'r')
        plt.title('Scatter epoch %d'%(epoch+1))
        #contour = plt.contour(X,Y,Z,1)
        #plt.clabel(contour,colors='k')
        # plt.show()

plt.figure(figsize=(10,4))
plt.plot(g_losses,label='Generator')
plt.plot(d_losses,label='Discriminator')
plt.legend()
plt.show()
