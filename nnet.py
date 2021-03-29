import numpy as np
import keras


class Sequential:

    def __init__(self, network):
        """
        Convert keras neural net to nnet.Sequential class
        :param network: keras neural network
        """

        self.input_shape = network.layers[0].get_weights()[0].shape[0]
        self.layers = []

        for i, layer in enumerate(network.layers):

            if isinstance(layer, keras.layers.core.Dense):
                #Add dense ReLU layer to instance
                assert( layer.activation == keras.activations.relu )

                weights, bias = layer.get_weights()
                layerclass = Dense(weights, bias, "l"+str(i))

                self.layers.append(layerclass)

    def generate_bounds(self, method, input, distance):
        """
        Generate upper and lower bounds on the affine function within each neuron

        :param method: method for bound propogation, 1 = interval arithmetic, 2 = DeepPoly
        :param input: numeric input to verify
        :param distance: l_inf distance around input to consider
        """

        #Input bounds are determined from l_inf norm around a numeric input
        input_ub = input + distance
        input_lb = input - distance

        if method == 1: #Interval arithmetic
            #Generate first layer bounds from
            self.layers[0].generate_interval_bounds(method, input_lb, input_ub)

            #Iteratively generate bounds on successive layers
            for i, layer in enumerate(self.layers[1:],start=1):
                prev_layer = self.layers[i-1]
                self.layers[i].generate_interval_bounds(prev_layer.relu_lb, prev_layer.relu_ub)

        if method == 2: #DeepPoly

            for i, layer in enumerate(self.layers):



    def backwards_pass(self, neuron):

        pass



class Layer:

    def __init__(self, weights, bias, label):

        self.weights = weights
        self.bias = bias
        self.label = label


class Dense(Layer):

    def __init__(self, weights, bias, label):

        super().__init__(weights,bias,label)
        self.input_shape = self.weights.shape[0]
        self.output_shape = self.weights.shape[1]

    def generate_interval_bounds(self, prev_l, prev_u):
        """
        Generate bounds for the layer using interval arithmetic given numeric bounds on the previous layer
        :param prev_l: lower bound on previous layer
        :param prev_u: upper bound on previous layer
        """


        uweight = np.where(self.weights > 0, self.weights, 0) #Positive weight matrix
        lweight = np.where(self.weights < 0, self.weights, 0) #Negative weight matrix

        #Bounds on affine function
        self.aff_ub = np.matmul(uweight.T, prev_u) + np.matmul(lweight.T, prev_l) + self.bias
        self.aff_lb = np.matmul(uweight.T, prev_l) + np.matmul(lweight.T, prev_u) + self.bias

        #Post-activation bounds on ReLU
        self.relu_ub = np.maximum(self.aff_ub, 0)
        self.relu_lb = np.maximum(self.aff_lb, 0)



class AffineFunction:

    def __init__(self, w = {}, b = 0):

        self.w = {}
        for key, val in w.items():
            if val != 0:
                self.w[key] = val

        self.b = b

    def add(self, func, coeff = 1):

        assert (isinstance(func,AffineFunction))
        for key, val in func.w:
            self.w[key] = self.w.get(key,0) + coeff*val

        self.b = self.b + coeff*func.b

    def remove_term(self,i):

        self.w.pop(i)

    def maxindex(self):

        return max(list(self.w.keys()).append(0))