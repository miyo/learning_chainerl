# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from chainer import training
from chainer.training import extensions

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            #self.conv1 = L.Convolution2D(1, 16, 3, 1, 1) # input-ch=1, output-ch=16, filter-size=3, stride=1, padding=1, 
            self.conv1 = L.Convolution2D(in_channels=1,
                                         out_channels=16,
                                         ksize=3,
                                         stride=1,
                                         pad=1)
            self.conv2 = L.Convolution2D(in_channels=16,
                                         out_channels=64,
                                         ksize=3,
                                         stride=1,
                                         pad=1)
            self.l3 = L.Linear(256, 10)  # classify
                                         # conv1: 8x8 -> 8x8
                                         # max_pooling_2d(1): 8x8 -> 4x4
                                         # conv2: 4x4 -> 4x4
                                         # max_pooling_2d(2): 4x4 -> 2x2
                                         # 2x2 * 64ch = 256
    def __call__(self, x):
        #h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2) # 2x2 max-pooling, stride=2
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)),
                              ksize=2,
                              stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)),
                              ksize=2,
                              stride=2)
        y = self.l3(h2)
        return y

epoch = 1000
batchsize = 100

# data creation
digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
data_train = (data_train).astype(np.float32)
data_test = (data_test).astype(np.float32)
data_train = data_train.reshape((len(data_train), 1, 8, 8)) # 64 -> 1x8x8, (color x height x width)
data_test = data_test.reshape((len(data_test), 1, 8, 8))

train = chainer.datasets.TupleDataset(data_train, label_train)
test = chainer.datasets.TupleDataset(data_test, label_test)

# setup neural network
model = L.Classifier(MyChain(), lossfun = F.softmax_cross_entropy)
# chainer.serializers.load_npz('result/out.model', model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# iterator definition
train_iter = chainer.iterators.SerialIterator(train, batchsize) # learning
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False) # testing

# updater registration
updater = training.StandardUpdater(train_iter, optimizer)

# trainer registration
trainer = training.Trainer(updater, (epoch, 'epoch'))

# print and save status
trainer.extend(extensions.LogReport()) # log
trainer.extend(extensions.Evaluator(test_iter, model)) # print #. of epochs
trainer.extend(extensions.PrintReport(['epoch',
                                       'main/loss',
                                       'validation/main/loss',
                                       'main/accuracy',
                                       'validation/main/accuracy',
                                       'elapsed_time'])) # print status
trainer.extend(extensions.dump_graph('main/loss')) # neural network structure
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png')) # accuracy graph
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png')) # accuracy graph
#trainer.extend(extensions.snapshot(), trigger=(100, 'epoch')) # save a snapshot
#trainer.serializers.load_npz('result/snapshot_iter_500', trainer) # to restart
chainer.serializers.save_npz('result/out.model', model)

# start
trainer.run()


                                       
