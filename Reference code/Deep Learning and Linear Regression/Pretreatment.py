from torch import nn
import torch

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("./dataset/Income1.csv")
X=torch.from_numpy(data.Education.values.reshape(-1,1).astype(np.float64))
print(X)

Y=torch.from_numpy(data.Income.values.reshape(-1,1).astype(np.float64))
print(Y)

model=nn.Linear(1,1)

#计算均方误差
loss_fn=nn.MSELoss()   #损失函数
#优化算法
opt=torch.optim.SGD(model.parameters(),lr=0.0001)












