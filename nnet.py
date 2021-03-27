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

        for i in range(len(network.layers)):
            layer = network.layers[i]
            if isinstance(layer, keras.layers.core.Dense):
                #Add dense ReLU layer to instance
                assert( layer.activation == keras.activations.relu )

                weights, bias = layer.get_weights()
                layerclass = Dense(weights, bias, "l"+str(i))

                self.layers.append(layerclass)

    def generate_bounds(self, method, input, distance):
        """
        Generate upper and lower bounds on the affine function within each neuron

        :param method: method for bound propogation, 1 = interval arithmetic
        :param input: numeric input to verify
        :param distance: l_inf distance around input to consider
        """

        #Input bounds are determined from l_inf norm around a numeric input
        input_ub = input + distance
        input_lb = input - distance

        #Generate first layer bounds from
        self.layers[0].generate_bounds(method, input_lb, input_ub)

        #Iteratively generate bounds on successive layers
        for i in range(1,len(self.layers)):
            prev_layer = self.layers[i-1]
            self.layers[i].generate_bounds(method, prev_layer.relu_lb, prev_layer.relu_ub)




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

    def generate_bounds(self, method, prev_l, prev_u):
        """
        Generate bounds for the layer given numeric bounds on the previous layer
        :param method: 1 = Interval Arithmetic
        :param prev_l: lower bound on previous layer
        :param prev_u: upper bound on previous layer
        """

        if method == 1: #Interval arithmetic

            uweight = np.where(self.weights > 0, self.weights, 0) #Positive weight matrix
            lweight = np.where(self.weights < 0, self.weights, 0) #Negative weight matrix

            #Bounds on affine function
            self.aff_ub = np.matmul(uweight.T, prev_u) + np.matmul(lweight.T, prev_l) + self.bias
            self.aff_lb = np.matmul(uweight.T, prev_l) + np.matmul(lweight.T, prev_u) + self.bias

            #Post-activation bounds on ReLU
            self.relu_ub = np.maximum(self.aff_ub, 0)
            self.relu_lb = np.maximum(self.aff_lb, 0)



