from mxnet import gluon,nd,init
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Dense(256,activation='relu')
        self.output = nn.Dense(10)

    def forward(self,x):
        return self.output(self.hidden(x))



class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential,self).__init__(**kwargs)

    def add(self,block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)

        return x




X = nd.random.normal(shape=(2,40))
net = MySequential()
net.add(nn.Dense(256,activation='relu'))
net.add(nn.Dense(10))
net.initialize()
# print(X)S
# print(net(X))


##### get params


net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'))
net.add(nn.Dense(10))
net.initialize()

net(X)

print(net[0].params)

print(net[0].params['dense2_weight'])
print(net[0].weight.data)   ## data is a method , so we need to use data() to get the data, ndarray can use grad not need ()
print(net[0].weight.data())
print(net.collect_params())
print(net.collect_params('.*weight'))

net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])


#### your own initial class

class MyInit(init.Initializer):
    def _init_weight(self,name,data):
        data[:] = nd.random.uniform(low=-10,high=10,shape=data.shape)
        data *= data.abs() > 5


net[0].weight.initialize(init=MyInit(), force_reinit=True)
print(net[0].weight.data()[0])
