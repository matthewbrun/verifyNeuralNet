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

        def generate_bounds(self, method):
            """

            :param self:
            :param method: 1 = IntervalArithmetic
            :return:
            """

            for layer in range(len(self.layers)):

                layer.generate_bounds(method)




class Layer:

    def __init__(self, weights, bias):

        self.weights = weights
        self.bias = bias

    def generate_bounds(self, method):

        pass

class Dense(Layer):

    def generate_bounds(self, method, prev_weights, prev_l, prev_u):
        """

        :param method: 1 = Interval Arithmetic
        :return:
        """

        if method == 1:
