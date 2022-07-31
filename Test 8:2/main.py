import keras
from keras import layers

import nnet
import numpy as np
import verifyNeuralNet as vnn
import train



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

def test_on_trained():

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.expand_dims(x_test, -1)
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential()
    model.add(layers.Dense(16, input_dim=28*28, activation='relu'))
    #model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Activation(activation='softmax'))

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


    real_class = 0
    compare_class = 8
    sample_i = np.random.choice(np.where(y_test[:,real_class]==1)[0])
    sample = x_test[sample_i, :]

    score_arr = np.zeros(5)
    methods = ["IntervalArithmetic", "DeepPoly", "FastC2V", "FlatC2V", "MeanC2V"]
    for i in range(5):
        options = nnet.BoundsOptions(methods[i], use_viol = True)
        score = vnn.boundDiff(model, sample, .2, real_class, compare_class, options)
        score_arr[i] = score
        print("BoundDiff ", methods[i], ": ", score)

    return score_arr

def run_multiple(iter):

    scores = np.empty((0,5))
    for i in range(iter):

        score_arr = test_on_trained()
        scores = np.append(scores, np.reshape(score_arr,(1,5)), axis=0)
        print(scores)
        np.save("scores.npy", scores)


#test_deeppoly_bounds()

#test_LP()

#test_layer_bound_solution()

#test_bound_quality()

#test_flatcut()

run_multiple(10)