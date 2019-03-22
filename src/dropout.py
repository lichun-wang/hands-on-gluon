from mxnet import gluon
from mxnet import autograd,nd


def dropout(X, dropout_prob):
    keep_prob = 1 - dropout_prob
    mask = nd.random.uniform(0,1,X.shape) < keep_prob
    return X * mask

a = nd.random.normal(shape=(3,4))
b = dropout(a,0.5)
print(a)
print(b)