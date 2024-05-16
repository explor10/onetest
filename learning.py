#%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,)).numpy()
time.numpy()
plt.plot(time,x)
plt.xlabel('time')
plt.ylabel('x')
plt.show()
