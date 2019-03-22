import os,sys
import mxnet as mx
from mxnet import init ,nd, autograd, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn


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

def Lenet():
    net = nn.Sequential()
    net.add(
        nn.Conv2D(kernel_size=5,channels=6,activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(kernel_size=5,channels=16, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2), ### strides not stride
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10)
    )

    return net


def vgg_block(num_conv, num_ch):
    blk = nn.Sequential()
    for _ in range(num_conv):
        blk.add(nn.Conv2D(channels=num_ch,kernel_size=3,padding=1,activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


def vgg(conv_arch):
    blk = nn.Sequential()
    for (num_conv, num_ch) in conv_arch:
        blk.add(vgg_block(num_conv,num_ch))
    blk.add(
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10),
    )

    return blk

def evaluate_accuracy(data_iter, net, ctx):
    sum_acc, n = 0.0,0
    for x, y in data_iter:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        y_hat = net(x)
        sum_acc += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
        n += y.size
    return sum_acc / n

batch_size = 256
resize = 96
train_iter,test_iter = load_data_fashion_mnist(batch_size=batch_size,resize=resize)


# net = Lenet()


conv_arch = ((1,64),(1,128),(1,256),(2,512),(2,512))
net = vgg(conv_arch)

ctx = try_gpu()
net.initialize(init.Xavier(), ctx= ctx)  ## init function use normal is not good

lr = 0.05

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr,'wd':0.0009})
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

    print('epoch:{},acc:{},loss:{},test_acc:{}'.format(epoch, sum_acc/n, sum_loss/n, evaluate_accuracy(test_iter, net, ctx)))



