# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from chainer import training
from chainer.training import extensions

# 1-hidden layer
class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 3) # 2-input, 3-output
            self.l2 = L.Linear(3, 2) # 3-input, 2-output
    def __call__(self, x):
        h1 = F.relu(self.l1(x)) # ReLU
        y = self.l2(h1)
        return y

# perceptron
class MyChain2(chainer.Chain):
    def __init__(self):
        super(MyChain2, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 2) # 2-input, 2-output
    def __call__(self, x):
        y = self.l1(x)
        return y
    
# 5-layers(with 3-hidden layers)
class MyChain3(chainer.Chain):
    def __init__(self):
        super(MyChain3, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 6) # 2-input, 6-output
            self.l2 = L.Linear(6, 3) # 6-input, 3-output
            self.l3 = L.Linear(3, 5) # 3-input, 5-output
            self.l4 = L.Linear(5, 2) # 5-input, 2-output
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)
        return y

test = np.array(([0,0], [0,1], [1,0], [1,1], [0.7,0.8], [0.2,0.4], [0.9,0.2]), dtype=np.float32)
model = L.Classifier(MyChain(), lossfun = F.softmax_cross_entropy)
#model = L.Classifier(MyChain3(), lossfun = F.softmax_cross_entropy)
chainer.serializers.load_npz('result/or_out.model', model)

print(model.predictor.l1.W.data)
print(model.predictor.l1.b.data)
print(model.predictor.l2.W.data)
print(model.predictor.l2.b.data)
