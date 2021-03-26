import numpy as np
import keras


class Sequential:

    def __init__(self, network):

        self.layers = []
        for layer in network.layers:
            if isinstance(layer, keras.layers.core.Dense):
                assert( layer.activation == keras.activations.relu )

                weights, bias = layer.get_weights()
                layerclass = Dense(weights, bias)

                self.layers.append(layerclass)

    def generate_bounds(self, method, input, distance):
        """

        :param method: method for bound propogation, 1 = interval arithmetic
        :param input: numeric input to verify
        :param distance: l_inf distance around input to consider
        :return:
        """
        input_u = input + distance
        input_l = input - distance
        self.layers[0].generate_bounds(method, input_l, input_u)

        for i in range(1,len(self.layers)):
            prev_layer = self.layers[i-1]
            self.layers[i].generate_bounds(method, prev_layer.lb, prev_layer.ub)




class Layer:

    def __init__(self, weights, bias):

        self.weights = weights
        self.bias = bias

    def generate_bounds(self, method):

        pass

class Dense(Layer):

    def generate_bounds(self, method, prev_l, prev_u):
        """

        :param method: 1 = Interval Arithmetic
        :return:
        """

        if method == 1:

            uweight = np.where(self.weights > 0, self.weights, 0)
            lweight = np.where(self.weights < 0, self.weights, 0)

            self.ub = np.matmul(uweight.T, prev_u) + np.matmul(lweight.T, prev_l) + self.bias
            self.lb = np.matmul(uweight.T, prev_l) + np.matmul(lweight.T, prev_u) + self.bias


