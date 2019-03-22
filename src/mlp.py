from mxnet import nd,autograd
from mxnet import gluon
from mxnet.gluon import data as gdata
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



input_num = 28*28
output_num = 10

hidden_num = 256

w1 = nd.random.normal(scale=0.01, shape=(input_num,hidden_num))
b1 = nd.zeros(hidden_num)

w2 = nd.random.normal(scale=0.01, shape=(hidden_num, output_num))
b2 = nd.zeros(output_num)

w1.attach_grad()
b1.attach_grad()
w2.attach_grad()
b2.attach_grad()


def ReLU(X):
    return nd.maximum(X, 0)

def net(X):
    hid =   ReLU(nd.dot(X.reshape((-1,input_num)),w1) + b1)
    return nd.dot(hid,w2) + b2


def sgd(params,lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size



loss = gloss.SoftmaxCELoss()  ## contain softmax + crossentropy

batch_size = 128
train_iter,test_iter = load_data_fashion_mnist(batch_size=batch_size)
epoch_num = 10
lr = 0.05

for epoch in range(epoch_num):
    sum_loss, sum_acc, n = 0.0, 0.0, 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat,y)
        l.backward()
        sgd([w1,b1,w2,b2], lr=lr,batch_size=batch_size)

        sum_loss += l.sum().asscalar()
        sum_acc += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()  ### argmax not nd.argmax
        n += batch_size

    print('epoch:{},acc;{},loss:{}'.format(epoch,sum_acc/n,sum_loss/n))






