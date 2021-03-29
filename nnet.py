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
            self.layers[0].generate_interval_bounds(input_lb, input_ub)

            #Iteratively generate bounds on successive layers
            for i, layer in enumerate(self.layers[1:],start=1):
                prev_layer = self.layers[i-1]
                self.layers[i].generate_interval_bounds(prev_layer.numeric_relu_lb, prev_layer.numeric_relu_ub)

        if method == 2: #DeepPoly
            # Iteratively generate bounds on successive layers
            for i, layer in enumerate(self.layers):
                #Generate lower and upper bounds from a backwards pass
                L, U = self.backwards_pass(i, input_lb, input_ub)

                #Set layer numeric lower and upper affine bounds
                layer.numeric_aff_ub = U
                layer.numeric_aff_lb = L

                #Generate lower and upper bounding affine function for layer

                ub_w = layer.weights #upper bounding weights matrix
                lb_w = layer.weights #lower bounding weights matrix

                ub_b = layer.bias #upper bounding intercept
                lb_b = layer.bias #lower bounding intercept

                for i in range(len(U)):

                    #If U <= 0, inactive, set bounding functions to 0
                    if U[i] <= 0:
                        ub_w[:,i] = 0
                        lb_w[:,i] = 0

                        ub_b[i] = 0
                        lb_b[i] = 0

                    #If U > 0 and L < 0, use DeepPoly bounding functions
                    elif L[i] < 0:

                        ub_w[:,i] = U[i]/(U[i]-L[i]) * ub_w[:,i]
                        ub_b[i] = U[i]/(U[i]-L[i]) * (ub_b[i] - L[i])

                        if abs(L[i]) >= abs(U[i]):

                            lb_w[:,i] = 0
                            lb_b[i] = 0

                #Add bounding function parameters to layer object
                layer.funcw_aff_ub = ub_w
                layer.funcw_aff_lb = lb_w

                layer.funcb_aff_ub = ub_b
                layer.funcb_aff_lb = lb_b







    def backwards_pass(self, neuron, input_lb, input_ub):

        #Get current neuron affine functions
        c_uw = self.layers[neuron].weights #positive for upper bound
        c_ub = self.layers[neuron].bias

        c_lw = -self.layers[neuron].weights #negative for lower bound
        c_lb = -self.layers[neuron].bias

        #Iterate backwards over layers
        for i in range(neuron-1,-1,-1):

            cuwp = np.where(c_uw > 0, c_uw, 0) #postive weight matrix for upper bound
            cuwn = np.where(c_uw < 0, c_uw, 0) #negative weight matrix for upper bound

            clwp = np.where(c_lw > 0, c_lw, 0) #positive weight matrix for lower bound
            clwn = np.where(c_lw < 0, c_lw, 0) #negative weight matrix for lower bound

            #Transform weights/bias backwards through layer by taking affine composite
            c_uw = np.matmul(self.layers[i].funcw_aff_ub, cuwp) + np.matmul(self.layers[i].funcw_aff_lb, cuwn)
            c_lw = np.matmul(self.layers[i].funcw_aff_ub, clwp) + np.matmul(self.layers[i].funcw_aff_lb, clwn)

            c_ub = c_ub + np.matmul(self.layers[i].funcb_aff_ub, cuwp) + np.matmul(self.layers[i].funcb_aff_lb, cuwn)
            c_lb = c_lb + np.matmul(self.layers[i].funcb_aff_ub, clwp) + np.matmul(self.layers[i].funcb_aff_lb, clwn)

        #Maximize upper/lower bound over input space
        cuwp = np.where(c_uw > 0, c_uw, 0)
        cuwn = np.where(c_uw < 0, c_uw, 0)

        clwp = np.where(c_lw > 0, c_lw, 0)
        clwn = np.where(c_lw < 0, c_lw, 0)

        ub = np.matmul(cuwp.T, input_ub) + np.matmul(cuwn.T, input_lb) + c_ub
        lb = -1 * (np.matmul(clwp.T, input_ub) + np.matmul(clwn.T, input_lb) + c_lb)

        return lb, ub


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
        self.numeric_aff_ub = np.matmul(uweight.T, prev_u) + np.matmul(lweight.T, prev_l) + self.bias
        self.numeric_aff_lb = np.matmul(uweight.T, prev_l) + np.matmul(lweight.T, prev_u) + self.bias

        #Post-activation bounds on ReLU
        self.numeric_relu_ub = np.maximum(self.numeric_aff_ub, 0)
        self.numeric_relu_lb = np.maximum(self.numeric_aff_lb, 0)



class AffineFunction:

    def __init__(self, w = [], b = 0):

        self.w = {}
        for i, val in enumerate(w):
            if val != 0:
                self.w[i] = val

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