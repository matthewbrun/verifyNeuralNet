import keras
from keras import layers

import nnet
import numpy as np



def train_model():

    model = keras.Sequential()
    model.add(layers.Dense(4, input_dim=8, activation='relu'))
    model.add(layers.Dense(2,activation='relu'))

    return model


def test_bounds():

    m = train_model()
    n = nnet.Sequential(m)

    inp = np.arange(8)
    d = .5

    n.generate_bounds(1, inp, d)

    return n
