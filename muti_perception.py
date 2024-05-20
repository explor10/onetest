import torch
from torch import nn
import matplotlib
from d2l import torch as d2l
from matplotlib import pyplot as plt
x=torch.arange(-8,8,0.1,dtype=torch.float32,requires_grad=True)
y=torch.relu(x)
y.backward(torch.ones_like(x),retain_graph=True)
x.grad.data.zero_()#梯度重置与 zero_grad区分

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens=784,10,256

W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[W1,b1,W2,b2]

def relu(x):
    a=torch.zeros_like(x)
    return torch.max(x,a)

def net(X):
    X=X.reshape((-1,num_inputs))
    H=relu(X@W1+b1)
    return (H@W2+b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater=torch.optim.SGD(params,lr=lr)
def train_batch(net, X, y, loss, trainer):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    #net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.mean().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        num_batches=len(train_iter)
        print(num_batches)
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net, features, labels, loss, updater)
            print(l,acc)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
print(f'loss {metric[0] / metric[2]:.3f}, train acc '
      f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')

