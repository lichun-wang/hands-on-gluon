from mxnet import nd,autograd,init
from mxnet.gluon import nn

class MyDense(nn.Block):
    def __init__(self,units,in_units,**kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units,units))  ## will create weight
        self.bias = self.params.get('bias',shape=(units,))

    def forward(self,X):
        linear = nd.dot(X,self.weight.data()) + self.bias.data()
        return nd.relu(linear)



net = MyDense(10,200)
net.initialize(init.Normal(sigma=0.01))


a = nd.random.normal(shape=(5,200))
print(net(a))
print(net.weight.data())

