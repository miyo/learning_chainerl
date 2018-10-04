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
            self.l1 = L.Linear(64, 100)  # 64-input, 100-output
            self.l2 = L.Linear(100, 100) # 100-input, 100-output
            self.l3 = L.Linear(100, 10)  # 100-input, 10-output
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

epoch = 1000
batchsize = 100

# data creation
digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
data_train = (data_train).astype(np.float32)
data_test = (data_test).astype(np.float32)

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
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png')) # accuracy graph
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png')) # accuracy graph
#trainer.extend(extensions.snapshot(), trigger=(100, 'epoch')) # save a snapshot
#trainer.serializers.load_npz('result/snapshot_iter_500', trainer) # to restart
#chainer.serializers.save_npz('result/out.model', model)

# start
trainer.run()


                                       
