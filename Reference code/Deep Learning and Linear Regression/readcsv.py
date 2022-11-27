import torch

import pandas as pd
import numpy
import matplotlib.pyplot as plt

data=pd.read_csv("./dataset/Income1.csv")
# data.info()  #返回这个文件的一些信息
print(data)
plt.scatter(data.Education,data.Income)
plt.xlabel("Education"),plt.ylabel("Income")
plt.show()