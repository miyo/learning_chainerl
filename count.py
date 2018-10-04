# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(3, 6) # 2-input, 6-output
            self.l2 = L.Linear(6, 6) # 6-input, 3-output
            self.l3 = L.Linear(6, 4) # 3-input, 5-output
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

epoch = 200
batchsize = 1

# data creation
trainx = np.array(([0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=np.float32)
trainy = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32)
train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

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
#trainer.serializers.save_npz('result/out.model', model)

# start
trainer.run()


                                       
