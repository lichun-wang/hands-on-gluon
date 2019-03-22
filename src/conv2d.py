from mxnet import nd
from mxnet.gluon import nn


def corr2d(X,K):
    h, w = K.shape
    Y = nd.zeros(shape=(X.shape[0]-h + 1, X.shape[1]-w+1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()

    return Y

class Conv2d(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2d,self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))   ### important self.bias shape = 1

    def forward(self,X):
        return corr2d(X,self.weight.data()) + self.bias.data()


X = nd.random.uniform(shape=(4,4))
K = nd.random.uniform(shape=(2,2))

net = Conv2d(kernel_size=K.shape)
net.initialize()

Y = net(X)

print(X)
print(K)
print(Y)