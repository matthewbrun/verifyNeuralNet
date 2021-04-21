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

    print(vnn.bigMLP(m, inp, d, 0, 1, 2))


def test_deeppoly_bounds():

    m = train_model()
    n = nnet.Sequential(m)
    n2 = nnet.Sequential(m)

    inp = np.arange(24)
    d = .5

    n.generate_bounds(2, inp, d)

    n2.generate_bounds(1, inp, d)

    print("\n\n\n")
    for layer in n2.layers:
        print(layer.numeric_aff_lb)
        print(layer.numeric_aff_ub)

def test_layer_bound_solution():

    m = train_model()
    inp = np.arange(24)

    out = m.predict(np.array([inp]))
    d = .5

    #print(out[0][1]-out[0][0])

    #print(vnn.boundDiff(m, inp, d, 0, 1, 1))

    #print(vnn.boundDiff(m, inp, d, 0, 1, 2))

    print(vnn.boundDiff(m, inp, d, 0, 1, 3))


#test_deeppoly_bounds()

#test_LP()

test_layer_bound_solution()