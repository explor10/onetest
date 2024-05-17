import torch
#关于 tensor
x=torch.arange(12)#有序张量 int 型
print(x)
print(x.shape)#torch.Size([12])
print(x.size())#以上等同
print(x.numel())#一共有多少元素
print(x.reshape(3,-1))#-1表示自动计算维度
a=torch.arange(12,dtype=torch.float32).reshape(3,-1)
b=torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((a,b),dim=0))
print(torch.cat((a,b),dim=1))#cat用来连接两个 tensor
print(a==b)#返回值为 bool类型
print(a>b)#返回一个布尔类型矩阵
x.sum()#求和产生一个单元素张量
#tensor的索引用中括号，:表示沿着这个轴所有的元素
A=a.numpy()#转换为ndarray
B=torch.tensor(A)#ndarray转换为 tensor
c=torch.tensor([3])
c=c.item()#单个 tensor 转换为标量

'''数据处理'''

import os
import pandas as pd
os.makedirs(os.path.join('..','data'),exist_ok=True)#创建文件夹
data_file= os.path.join('..','data','house_tiny.csv')#文件路径
with open(data_file,'w') as f:
    f.write('numrooms,alley,price\n')#列名
    f.write('NA,Pave,12500\n')
    f.write('2,NA,10000\n')
    f.write('4,NA,16000\n')
    f.write('NA,NA,20000\n')
data=pd.read_csv(data_file)
print(data)
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]#iloc是位置索引，切片
idx=inputs.isna().sum(axis=0).idxmax()
print(idx)
inputs=inputs.drop(idx,axis=1)
print(inputs)
inputs=inputs.fillna(inputs.mean(numeric_only=True))#缺失值补充用平均值
#inputs=pd.get_dummies(inputs,dummy_na=True)
X=torch.tensor(inputs.to_numpy(dtype=float))
print(X,'\n',len(X))

'''tensor的向量运算'''

#tensor的任意一个元素也是 tensor
t=torch.arange(4,dtype=torch.float32)
print(t[3],len(t),t.shape)#shape是维度
p=torch.arange(24).reshape(2,3,4)
print(len(p),p.sum(axis=[0]).shape[0],p/p.sum(axis=[1],keepdims=True))#sum之后是个标量
q=torch.ones(4,dtype=torch.float32)
print(torch.dot(q,t))#向量点乘，*是叉乘
#torch.mv(a,b)是一个矩阵-向量积，a 的列数和b的长度必须相同
#torch.mm(A,B)是一个矩阵-矩阵乘积，注意对齐
#torch.norm(A)向量 A 的 L2 范数
#torch.abs(A).sum()L1 范数


'''calculus'''
der=torch.arange(12,dtype=torch.float32,requires_grad=True)
#der2=torch.dot(der,der).backward()
#print(der2,der.grad)
#der.grad.zero_()
der1=der*der
u=der1.detach()#分离计算，这个 U 看做一个常数，不是关于 der 的函数
z1=(u*der).sum().backward()#求导是 u，而不是 3der^2
print(der.grad)
def f(A):#控制流
    b=A*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a1=torch.randn(3,requires_grad=True)
d=f(a1)
d.sum().backward()
print(a1,a1.grad)

'''查找一个 module 里所有的函数和类'''
print(dir(torch))
import shap
print(dir(shap))
#help(shap.ActionOptimizer)