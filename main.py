import keras
from keras import layers

import nnet
import numpy as np
import verifyNeuralNet as vnn



def train_model():

    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=24, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(2,activation='relu'))

    return model


def test_bounds():

    m = train_model()
    n = nnet.Sequential(m)

    inp = np.arange(8)
    d = .5

    n.generate_bounds(1, inp, d)

    return n

def test_LP():
    m = train_model()
    inp = np.arange(24)
    d = .5

    print(vnn.bigMLP(m, inp, d, 0, 1, 1))

test_LP()