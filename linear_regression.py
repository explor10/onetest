import math
import time
import numpy as np
import torch
from torch.utils import data
from torch import nn

#定义正太分布函数
def normal(x,mu,sigma):
    return 1/math.sqrt(2*math.pi*sigma**2)*(np.exp(-0.5*(x-mu)**2/sigma**2))
x=np.arange(-7,7,0.01)

def syn_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x, y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=syn_data(true_w,true_b,1001)

def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
#help(data.TensorDataset)
#help(data.DataLoader)

batch_size=10
data_iter=load_array((features,labels),batch_size)
print(next(iter(data_iter)))

net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0)
net[0].bias.data.fill_(0)

loss=nn.HuberLoss(reduction='mean')

trainer=torch.optim.SGD(net.parameters(),lr=0.03)
num_epochs=5
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y) 
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch{epoch+1},loss{l:f}')

w= net[0].weight.data
b=net[0].bias.data
print(w,b,true_w-w.reshape(true_w.shape),true_b-b)