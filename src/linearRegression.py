from mxnet import autograd
from mxnet import nd
from matplotlib import pyplot as plt
import d2lzh
import random

num_input = 2
num_examples = 1000
true_w = [2,-3]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_input))
labels = true_w[0]* features[:,0] + true_w[1]* features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape = labels.shape)

d2lzh.set_figsize()
plt.scatter(features[:,1].asnumpy(), labels.asnumpy(),1);


def data_iter(features, labels, batch_size):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  ###take function and yield function

batch_size = 10
iter = data_iter(features,labels,batch_size)
# print(next(iter))
# print(next(iter))  ## next function can get next iter by the iter object

w = nd.random.normal(scale=0.01,shape=(num_input,1))
b = nd.zeros((1,))
w.attach_grad()
b.attach_grad()

def linearReg(X,w,b):
    return nd.dot(X,w) + b

def squared_loss(y_true,y):
    return (y_true - y.reshape(y_true.shape)) ** 2 / 2 ## why this need to reshape, because dim is not same

def sgd(params, lr, batch_size):
    for param in params:
        param[:] =param - lr * param.grad / batch_size


lr = 0.03
num_epoch = 100
net = linearReg
loss = squared_loss

for epoch in range(num_epoch):
    for X,y in data_iter(features,labels,10):
        with autograd.record():
            l = loss(net(X,w,b),y)
        l.backward()
        sgd([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print('epoch:{},loss:{}'.format(epoch,train_l))


print(w)
print(b)


