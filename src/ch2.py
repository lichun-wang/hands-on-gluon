import os
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet import autograd
print('mxnet version:{}'.format(mx.__version__))

x = nd.arange(12)

print(x)

print(x.shape)

print(x.size)

print(x.reshape((3,4)))

zero = nd.zeros((2,3,4))
one = nd.ones((2,3,4))

print(zero)
print(one)

## creat by list

Y = nd.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(Y)

### creat random

S = nd.random.normal(0,1,shape=(3,4))
print(S)


print(S*Y)

print(S/Y)

print(S.exp())

print(nd.dot(S,Y.T))

print(nd.concat(S,Y,dim=0))
print(nd.concat(S,Y,dim=1))

print(S == Y)

print('S:L2:{}'.format(S.norm().asscalar()))

# broadcasting

A = nd.arange(3).reshape((3,1))
B = nd.arange(2).reshape((1,2))

print(A + B)


#ndarray -> numpy

C = S.asnumpy()
print(C)

# numpy -> ndarray

p = np.ones((2,3))
d = nd.array(p)
print(d)

#### cal gradient

x = nd.arange(4).reshape((4,1))
x.attach_grad()


with autograd.record():
    y = 2 * nd.dot(x.T,x)

y.backward()

print((x.grad - 4*x).norm().asscalar() == 0)

### mxnet







