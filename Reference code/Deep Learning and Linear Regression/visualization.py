import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import torch

data=pd.read_csv("./dataset/Income1.csv")
X = torch.from_numpy(data.Education.values.reshape(-1, 1)).type(torch.FloatTensor)
# print(X)

Y = torch.from_numpy(data.Income.values.reshape(-1, 1)).type(torch.FloatTensor)
# print(Y)

model=nn.Linear(1,1)

#计算均方误差
loss_fn=nn.MSELoss()   #损失函数
#优化算法
opt = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = model(x)   #使用模型预测
        loss=loss_fn(y,y_pred)  #根据预测结果计算损失
        opt.zero_grad()  #把变量的梯度清零
        loss.backward() #反向传播算法，求解梯度
        opt.step()  #优化模型参数
print(model.weight,model.bias)
plt.scatter(data.Education,data.Income)
plt.xlabel("Education"),plt.ylabel("Income")
plt.plot(X.numpy(),model(X).data.numpy(),c='r')
plt.show()