import torch
import torch.nn as nn
import numpy as np


data=np.loadtxt("./resource/german.data-numeric")
print(data)

h,w=data.shape
print(h,w)

for j in range(w-1):
    meanVal=np.mean(data[:,j])
    stdVal=np.std(data[:,j])
    data[:,j]=(data[:,j]-meanVal)/stdVal

np.random.shuffle(data)


train_data=data[:900,:w-1]
train_lab=data[:900,w-1]-1
test_data=data[900:,:w-1]
test_lab=data[900:,w-1]-1
# print(train_data[0],"\n", train_data)
# print(train_lab)


class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(24,2) # 由于24个维度已经固定了，所以这里写24
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)
        return out

def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())

net=LR()
criterion=nn.CrossEntropyLoss() # 使用CrossEntropyLoss损失
optm=torch.optim.Adam(net.parameters()) # Adam优化
epochs=1000 # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)
    loss=criterion(y_hat,y) # 计算损失
    optm.zero_grad() # 前一步的损失清零
    loss.backward() # 反向传播
    optm.step() # 优化
    if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))





