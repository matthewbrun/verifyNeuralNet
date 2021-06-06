import keras
from keras import layers

import nnet
import numpy as np
import verifyNeuralNet as vnn



def train_model():

    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=24, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
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

    options1 = nnet.BoundsOptions("IntervalArithmetic")
    options2 = nnet.BoundsOptions("DeepPoly")
    options3 = nnet.BoundsOptions("FastC2V", use_FP_relu = True)
    options4 = nnet.BoundsOptions("FlatC2V", use_FP_relu = True, use_viol = False, do_iterative_tighten = False,
                        use_flat_ubs = False)
    options5 = nnet.BoundsOptions("MeanC2V", num_points=100, use_FP_relu = True, use_viol = False,
                        use_mean_ubs = False)

    print(vnn.boundDiff(m, inp, d, 0, 1, options1))

    print(vnn.boundDiff(m, inp, d, 0, 1, options2))

    print(vnn.boundDiff(m, inp, d, 0, 1, options3))

    print(vnn.boundDiff(m, inp, d, 0, 1, options4))

    print(vnn.boundDiff(m, inp, d, 0, 1, options5))

def test_bound_quality():

    flag = True
    while flag:
        m = train_model()
        inp = np.arange(24)
        d = .5
        lb = inp - d
        ub = inp + d
        i = 0

        fcvb = vnn.boundDiff(m, inp, d, 0, 1, 3)

        for i in range(100):


            scale = (2*np.random.randint(0,2,24) - 1)*d

            out = m.predict(np.array([inp + scale]))

            if out[0][1]-out[0][0] > fcvb:
                print("WRONG")
                flag = False
                break

def test_flatcut():

    m = train_model()
    n = nnet.Sequential(m)
    inp = np.arange(24)
    d = .5
    lb = inp-d
    ub = inp+d

    n.layers[0].flattest_inequality(lb,ub,0,0)




#test_deeppoly_bounds()

#test_LP()

test_layer_bound_solution()

#test_bound_quality()

#test_flatcut()