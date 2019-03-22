import mxnet as mx
from mxnet import gluon,nd,autograd,init
from mxnet.gluon import nn
import time

net = nn.HybridSequential()    ###
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(500, activation='relu'),
        )

net.initialize(ctx=mx.gpu(0))
x = nd.random.normal(shape=(1000,500),ctx=mx.gpu(0))

net.hybridize()

t1 = time.time()
for _ in range(1000):
    x = net(x)
t2 = time.time()
print(t2-t1)
# net.export('model')


#### HybridBlock

class HybrideNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybrideNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # x = x.asnumpy()
        if 1 == 1:
            print(F)
        x = F.relu(self.hidden(x))
        return self.output(x)


net = HybrideNet()
net.initialize()

net.hybridize()

a = nd.random.normal(shape=(1, 1024))

net(a)