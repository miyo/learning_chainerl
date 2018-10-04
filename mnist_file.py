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

from PIL import Image

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

model = L.Classifier(MyChain(), lossfun = F.softmax_cross_entropy)
chainer.serializers.load_npz('result/out.model', model)

img = Image.open('test.png')
img = img.convert('L')
img = img.resize((8, 8))

img = 16.0 - np.asarray(img, dtype=np.float32)
img = img[np.newaxis, np.newaxis, :, :]
x = chainer.Variable(img)
y = model.predictor(x)
c = F.softmax(y).data.argmax()
print(c)
