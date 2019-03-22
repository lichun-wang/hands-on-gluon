import mxnet as mx
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet import nd, autograd, gluon, init
import os,sys


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    # root = os.path.expanduser(root)

    root = '../data/'
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.ones(shape=(1,),ctx=ctx)
    except:
        ctx = mx.cpu()

    return ctx

class Residual_block(nn.Block):
    def __init__(self, num_channels, use_1x1conv=True, strides=1, **kwargs):
        super(Residual_block,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels,kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self,X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(X + Y)


block = Residual_block(6,use_1x1conv=True)
block.initialize()
x = nd.random.normal(shape=(1,3,9,9))
x = block(x)
print(x.shape)

net = nn.Sequential()
net.add(nn.Conv2D(channels=64,kernel_size=7,padding=3,strides=2),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, padding=1, strides=2))

net.add(Residual_block(64,strides=2))
net.add(Residual_block(128,strides=2))
net.add(Residual_block(256,strides=2))
net.add(Residual_block(512,strides=2))

net.add(nn.GlobalAvgPool2D(),
        nn.Dense(10))



batch_size = 256
resize = 96
train_iter,test_iter = load_data_fashion_mnist(batch_size=batch_size,resize=resize)

ctx = try_gpu()
net.initialize(init.Xavier(), ctx= ctx)  ## init function use normal is not good

lr = 0.05

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr,'momentum':0.9,'wd':0.0009})
loss = gloss.SoftmaxCELoss()
epochs = 30

for epoch in range(epochs):
    sum_acc, sum_loss,n,index = 0.0, 0.0, 0, 0
    for x,y in train_iter:
        x,y = x.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():  ## be careful the record()
            y_hat = net(x)
            l = loss(y_hat,y)
        l.backward()
        trainer.step(batch_size)

        ls = l.mean().asscalar()
        ac = (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
        # print('epoch:{}, index:{}, acc:{}, loss:{}'.format(epoch,index, ac, ls))
        index += 1

        sum_loss += l.sum().asscalar()
        sum_acc += (y_hat.argmax(axis = 1) == y.astype('float32')).sum().asscalar()

        n += batch_size

    print('epoch:{},acc:{},loss:{}'.format(epoch, sum_acc/n, sum_loss/n))



