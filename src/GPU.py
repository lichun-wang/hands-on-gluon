import mxnet as mx
from mxnet import nd


a = nd.random.normal(shape=(2,3),ctx=mx.cpu())

b = a.copyto(mx.gpu(0))
c = a.as_in_context(mx.gpu(1))  ## if src and dst are in the same ctx,  as_in_context not copy but copyto will copy
print(a)
print(b)
print(c)