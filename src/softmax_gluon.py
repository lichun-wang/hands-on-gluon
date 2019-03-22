from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
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
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter

def evaluate_accuracy(data_iter,net):
    acc_sum , n = 0.0,0
    for X,y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).mean().asscalar()
        n += 1
    return acc_sum / n

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

Loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.05})  ## important

num_epoch = 10


for epoch in range(num_epoch):
    train_loss_sum , train_acc_sum = 0.0, 0.0
    n = 0
    for X,y in train_iter:
        with autograd.record():
            y_hat = net(X)
            loss = Loss(y_hat, y)
        loss.backward()

        trainer.step(batch_size)

        y = y.astype('float32')
        train_loss_sum += loss.sum().asscalar()
        train_acc_sum += ((y_hat.argmax(axis=1) == y).sum().asscalar())
        n += batch_size

    test_acc = evaluate_accuracy(test_iter,net)
    print('epoch:{},loss:{},train acc:{},test_acc:{}'.format(epoch,train_loss_sum/n, train_acc_sum/n, test_acc))
