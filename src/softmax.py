from mxnet import nd,autograd
from mxnet import gluon
from mxnet.gluon import data as gdata
import os,sys
import cv2



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


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)


input_num = 28*28
output_num = 10

w = nd.random.normal(scale=0.01,shape=(input_num,output_num))
b = nd.zeros(output_num)
print(b)

w.attach_grad()
b.attach_grad()

def softmax(X):
    X_exp = nd.exp(X)
    partition = X_exp.sum(axis=1,keepdims=True)
    return X_exp / partition

# X = nd.random.normal(scale=1,shape=(2,5))
# X_prob = softmax(X)
# print(X_prob)

def net(X):
    return softmax(nd.dot(X.reshape((-1,input_num)),w)+b)


def cross_entropy(y,y_true):
    return - nd.pick(y,y_true).log()


def accuracy(y,y_true):
    return (y.argmax(axis=1) == y_true.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter,net):
    acc_sum , n = 0.0,0
    for X,y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).mean().asscalar()
        n += 1
    return acc_sum / n


def sgd(params, lr, batch_size):
    for param in params:
        param[:] =param - lr * param.grad / batch_size





### train
epoch_num = 10
lr = 0.05
net = net

for epoch in range(epoch_num):
    train_loss_sum , train_acc_sum = 0.0, 0.0
    n = 0
    for X,y in train_iter:
        with autograd.record():
            y_hat = net(X)
            loss = cross_entropy(y_hat, y)
        loss.backward()

        sgd([w,b],lr,batch_size=batch_size)
        y = y.astype('float32')
        train_loss_sum += loss.sum().asscalar()
        train_acc_sum += ((y_hat.argmax(axis=1) == y).sum().asscalar())
        n += batch_size

    test_acc = evaluate_accuracy(test_iter,net)
    print('epoch:{},loss:{},train acc:{},test_acc:{}'.format(epoch,train_loss_sum/n, train_acc_sum/n, test_acc))
