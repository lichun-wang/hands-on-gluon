from mxnet import nd,autograd
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon





num_input = 2
num_examples = 1000
true_w = [2,-3]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_input))
labels = true_w[0]* features[:,0] + true_w[1]* features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape = labels.shape)


batch_size = 10

dataset = gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

# def net
net = nn.Sequential()
net.add(nn.Dense(1))

## init
net.initialize(init.Normal(sigma=0.01))

## loss
loss = gloss.L2Loss()


trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

num_epochs = 3

for epoch in range(num_epochs):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X),y).mean()
        l.backward()
        trainer.step(1)
    l = loss(net(features), labels)
    print('epoch:{},loss:{}'.format(epoch,l.mean().asnumpy()))


dense = net[0]
print(dense.weight.data())
print(dense.bias.data())