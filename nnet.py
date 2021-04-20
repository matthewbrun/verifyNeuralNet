import numpy as np
import keras

#TODO: reduce use of np.copy - only necessary when resulting object edited

class Sequential:

    def __init__(self, network):
        """
        Convert keras neural net to nnet.Sequential class
        :param network: keras neural network
        """

        self.input_shape = network.layers[0].get_weights()[0].shape[0]
        self.layers = []

        for layer in network.layers:

            if isinstance(layer, keras.layers.core.Dense):
                #Add dense ReLU layer to instance
                assert( layer.activation == keras.activations.relu )

                weights, bias = layer.get_weights()
                self.addLayer(weights, bias)

    def addLayer(self, weights, bias):

        layerclass = Dense(weights, bias, "l" + str(len(self.layers)))
        self.layers.append(layerclass)

    def generate_bounds(self, method, input, distance):
        """
        Generate upper and lower bounds on the affine function within each neuron

        :param method: method for bound propogation, 1 = interval arithmetic, 2 = DeepPoly, 3 = FastC2V
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
                self.layers[i].generate_interval_bounds(np.copy(prev_layer.numeric_relu_lb), np.copy(prev_layer.numeric_relu_ub))

        if method == 2: #DeepPoly

            #Keep affine bounding functions from previous layers in the form (weights, biases)
            aff_lbs = []
            aff_ubs = []

            # Iteratively generate bounds on successive layers

            for i, layer in enumerate(self.layers):
                #Generate lower and upper bounds from a backwards pass
                L, U, *__ = self.backwards_pass(i, aff_lbs, aff_ubs, input_lb, input_ub)

                #Set layer numeric lower and upper affine bounds
                layer.numeric_aff_ub = U
                layer.numeric_aff_lb = L

                #Generate lower and upper bounding affine function for layer

                layer.generate_DeepPoly_bounds()
                aff_lbs.append( [np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)] )
                aff_ubs.append( [np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)] )

        if method == 3: #FastC2V

            #Keep affine bounding functions from previous layers in the form (weights, biases)
            aff_lbs = []
            aff_ubs = []


            for i, layer in enumerate(self.layers):

                #Generate lower and upper bounds and solution data from a backwards pass
                L, U, x_lb, x_ub, top_lbs, top_ubs = self.backwards_pass(i, aff_lbs, aff_ubs, input_lb, input_ub)

                #Find solution from backwards pass data
                z_lb, z_ub = self.forwards_pass(i, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub)


                #This is inefficient: loops over each neuron in layer
                #is it possible to vectorize these computations?

                #Generate new tightened numeric bounds at each neuron in present layer
                for j in range(layer.output_shape):
                    new_aff_ubs_lb = []
                    new_aff_ubs_ub = []
                    for item in aff_ubs:
                        new_aff_ubs_lb.append( [np.copy(item[0]), np.copy(item[1])] ) #replaced affine upper bounds for neuron lower bound
                        new_aff_ubs_ub.append( [np.copy(item[0]), np.copy(item[1])] )  # replaced affine upper bounds for neuron upper bound

                    #Track numeric bounds on previous layer
                    prev_l = input_lb
                    prev_u = input_ub

                    #Tighten bounds on each preceeding layer
                    for k, tighten_layer in enumerate(self.layers[0:i]):

                        #Find most violated inequalities given solution for fixed neuron and lower/upper bounded solutions
                        new_aff_lb, viol_lb = tighten_layer.most_violated_inequality(j, prev_l, prev_u, z_lb[k][:,j], z_lb[k+1][:,j]) #lb

                        new_aff_ub, viol_ub = tighten_layer.most_violated_inequality(j, prev_l, prev_u, z_ub[k][:,j], z_ub[k + 1][:,j]) #ub

                        #If the inequalities are violated, replace for this neuron
                        new_aff_ubs_lb[k][0] = np.where(viol_lb > 0, new_aff_lb[0], new_aff_ubs_lb[k][0])
                        new_aff_ubs_ub[k][0] = np.where(viol_ub > 0, new_aff_ub[0], new_aff_ubs_lb[k][0])

                        new_aff_ubs_lb[k][1] = np.where(viol_lb > 0, new_aff_lb[1], new_aff_ubs_lb[k][1])
                        new_aff_ubs_ub[k][1] = np.where(viol_ub > 0, new_aff_ub[1], new_aff_ubs_lb[k][1])

                        # if viol_lb > 0:
                        #     new_aff_ubs_lb[k] = new_aff_lb
                        # if viol_ub > 0:
                        #     new_aff_ubs_ub[k] = new_aff_ub

                        #Update numeric bounds for next layer
                        prev_l = np.copy(tighten_layer.numeric_aff_lb)  #TODO: should these be RELU activated?
                        prev_u = np.copy(tighten_layer.numeric_aff_ub)

                    #This is inefficient: only need backwards pass to a single neuron in the last layer, not all neurons
                    #Also, only need lower or upper bound at a time, not both

                    #Repeat backwards pass with new affine bounds for current neuron
                    Lj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_lb, input_lb, input_ub)
                    __, Uj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_ub, input_lb, input_ub)

                    #Replace numeric bounds at current neuron if improved
                    L[j] = max(Lj_new[j], L[j])
                    U[j] = min(Uj_new[j], U[j])

                #Set layer numeric lower and upper affine bounds
                layer.numeric_aff_ub = U
                layer.numeric_aff_lb = L

                #Generate lower and upper bounding affine function for layer

                layer.generate_DeepPoly_bounds()
                aff_lbs.append( [np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)] )
                aff_ubs.append( [np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)] )











    def backwards_pass(self, l_num, aff_lbs, aff_ubs, input_lb, input_ub):
        """
        Generate upper and lower numeric bounds on layer affine outputs via a backwards pass over the network
        :param l_num: layer number on which to develop bounds
        :param aff_lbs: list of affine lower bounds on each preceeding layer, affine function defined by (weights, biases)
        :param aff_ubs: list of affine upper bounds on each preceeding layer, affine function defined by (weights, biases)
        :param input_lb: lower bounds on input space
        :param input_ub: upper bounds on input space
        :return: lb, ub: lower and upper numeric bounds on affine functions for layer
                x_lb, x_ub: feasible input points for lower and upper bounds at each neron in the layer
                top_lbs, top_ubs: boolean indicator of whether upper bounds were taken at each neuron (layer neurons down columns, input neurons on rows)
        """

        #Track if upper bounds are taken in back propogation
        top_lbs = []
        top_ubs = []

        #Get current neuron affine functions
        c_uw = np.copy(self.layers[l_num].weights) #positive for upper bound
        c_ub = np.copy(self.layers[l_num].bias)

        c_lw = -np.copy(self.layers[l_num].weights) #negative for lower bound
        c_lb = -np.copy(self.layers[l_num].bias)

        #Iterate backwards over layers
        for i in range(l_num-1,-1,-1):

            #Add upper bound indicators for current layer
            top_ubs.insert(0, c_uw > 0)
            top_lbs.insert(0, c_lw > 0)

            cuwp = np.where(c_uw > 0, c_uw, 0) #postive weight matrix for upper bound
            cuwn = np.where(c_uw < 0, c_uw, 0) #negative weight matrix for upper bound

            clwp = np.where(c_lw > 0, c_lw, 0) #positive weight matrix for lower bound
            clwn = np.where(c_lw < 0, c_lw, 0) #negative weight matrix for lower bound

            #Transform weights/bias backwards through layer by taking affine composite
            c_uw = np.matmul(aff_ubs[i][0], cuwp) + np.matmul(aff_lbs[i][0], cuwn)
            c_lw = np.matmul(aff_ubs[i][0], clwp) + np.matmul(aff_lbs[i][0], clwn)

            c_ub = c_ub + np.matmul(aff_ubs[i][1], cuwp) + np.matmul(aff_lbs[i][1], cuwn)
            c_lb = c_lb + np.matmul(aff_ubs[i][1], clwp) + np.matmul(aff_lbs[i][1], clwn)

        #Maximize upper/lower bound over input space
        cuwp = np.where(c_uw > 0, c_uw, 0)
        cuwn = np.where(c_uw < 0, c_uw, 0)

        clwp = np.where(c_lw > 0, c_lw, 0)
        clwn = np.where(c_lw < 0, c_lw, 0)

        ub = np.matmul(cuwp.T, input_ub) + np.matmul(cuwn.T, input_lb) + c_ub
        lb = -1 * (np.matmul(clwp.T, input_ub) + np.matmul(clwn.T, input_lb) + c_lb)

        #Generate feasible solution for upper and lower bounds
        x_ub = np.where(c_uw.T > 0, input_ub, input_lb).T #neuron in columns, x indices across rows
        x_lb = np.where(c_lw.T > 0, input_ub, input_lb).T

        return lb, ub, x_lb, x_ub, top_lbs, top_ubs



    def forwards_pass(self, l_num, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub):
        """
        Generates a full solution for preceeding layers using a forward pass, based on results from backward propogation
        :param l_num: layer number to which the forward pass is done
        :param aff_lbs: list of affine lower bounds on each preceeding layer, affine function defined by (weights, biases)
        :param aff_ubs: list of affine upper bounds on each preceeding layer, affine function defined by (weights, biases)
        :param top_lbs: list of boolean indicator of whether upper bounds were taken at each neuron within preceeding layers,
                        for each neuron in the current layer on the lower bound solution (list index for previous layer number,
                        rows of matrix for previous layer neurons, columns of matrix for present layer neurons)
        :param top_ubs: list of boolean indicator of whether upper bounds were taken at each neuron within preceeding layers,
                        for each neuron in the current layer on the upper bound solution
        :param x_lb: feasible input to generate layer's lower bound
        :param x_ub: feasible input to generate later's upper bound
        :return: z_lb, z_ub: lists of matrices with upper bound and lower bound solutions for each layer neuron (columns)
                            at preceeding layer (list index) neurons (row)
        """

        #Initialize solution with input values
        z_lb = [x_lb]
        z_ub = [x_ub]

        #Track solution in previous layer
        prev_lb = x_lb
        prev_ub = x_ub

        #Get ouput shape of current layer (columns in solution matrix z)
        n_neur = self.layers[l_num].output_shape

        #Forwards pass over preceeding layers
        for i in range(l_num):

            #Take lower bounding affine transformation on previous layer solution if upper bound not used,
            # otherwise take upper bounding affine transformation to get solution in next layer

            next_z_lb = (np.matmul(aff_lbs[i][0].T, prev_lb) + np.tile(np.reshape(aff_lbs[i][1], (-1,1)), (1,n_neur))) * (1 - top_lbs[i]) + \
                        (np.matmul(aff_ubs[i][0].T, prev_lb) + np.tile(np.reshape(aff_ubs[i][1], (-1,1)), (1,n_neur))) * top_lbs[i]
            next_z_ub = (np.matmul(aff_lbs[i][0].T, prev_ub) + np.tile(np.reshape(aff_lbs[i][1], (-1,1)), (1,n_neur))) * (1 - top_ubs[i]) + \
                        (np.matmul(aff_ubs[i][0].T, prev_lb) + np.tile(np.reshape(aff_ubs[i][1], (-1,1)), (1,n_neur))) * top_ubs[i]

            #Add layer solution to z vector
            z_lb.append(next_z_lb)
            z_ub.append(next_z_ub)

            #Update previous layer solution
            prev_lb = next_z_lb
            prev_ub = next_z_ub

            # print("Iter " + str(i))
            # print(prev_lb.shape)
            # print(top_lbs[i].shape)
            # print(aff_lbs[i][0].shape)

        return z_lb, z_ub




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

    def generate_DeepPoly_bounds(self):
        """
        Generate affine bounding functions on neuron layer using the DeepPoly method
        Requires self.numeric_aff_lb and self.numeric_aff_ub be set to numeric upper and lower bounds on each neruon
        """

        L = np.copy(self.numeric_aff_lb)
        U = np.copy(self.numeric_aff_ub)

        #TODO: delete?
        #Post-activation bounds on ReLU
        self.numeric_relu_ub = np.maximum(self.numeric_aff_ub, 0)
        self.numeric_relu_lb = np.maximum(self.numeric_aff_lb, 0)

        ub_w = np.copy(self.weights)  # upper bounding weights matrix
        lb_w = np.copy(self.weights)  # lower bounding weights matrix

        ub_b = np.copy(self.bias)  # upper bounding intercept
        lb_b = np.copy(self.bias)  # lower bounding intercept

        for i in range(len(U)):

            # If U <= 0, inactive, set bounding functions to 0
            if U[i] <= 0:
                ub_w[:, i] = 0
                lb_w[:, i] = 0

                ub_b[i] = 0
                lb_b[i] = 0

            # If U > 0 and L < 0, use DeepPoly bounding functions
            elif L[i] < 0:

                ub_w[:, i] = U[i] / (U[i] - L[i]) * ub_w[:, i]
                ub_b[i] = U[i] / (U[i] - L[i]) * (ub_b[i] - L[i])

                if abs(L[i]) >= abs(U[i]):
                    lb_w[:, i] = 0
                    lb_b[i] = 0

            # Otherwise, L >= 0, active, use default affine function as bounding functions

        # Add bounding function parameters to layer object
        self.funcw_aff_ub = ub_w
        self.funcw_aff_lb = lb_w

        self.funcb_aff_ub = ub_b
        self.funcb_aff_lb = lb_b

    def most_violated_inequality(self, neur, prev_l, prev_u, prev_z, curr_z):
        """
        Finds the affine representation of the most violated inequality from the convex relaxation for a neuron
        :param neur: neuron index number to tighten
        :param prev_l: numeric lower bounds on the previous layer
        :param prev_u: numeric upper bounds on the previous layer
        :param prev_z: forward pass solution in the previous layer for neuron
        :param curr_z: output of neuron
        :return: aff_fun, v = affine inequality function (weights, bias), violation
        """
        #need numeric bounds on previous layer, solution in previous layer

        #need information about layer output (x for layer) to determine violation?

        #remember to use negative affine function for lower bounds

        # aff_w = self.funcw_aff_ub
        # aff_b = self.funcb_aff_ub
        # v = np.ones((self.output_shape))
        #
        # aff_fun = [aff_w, aff_b]
        # return aff_fun, v

        aff_w = np.zeros(self.funcw_aff_ub.shape)
        aff_b = np.zeros(self.funcb_aff_ub.shape)

        v = np.zeros((self.output_shape))


        for i in range(self.output_shape):

            neur_weight = np.copy(self.weights[:,i])
            neur_bias = np.copy(self.bias[i])

            Lhat = np.where(neur_weight >= 0, prev_l, prev_u)
            Uhat = np.where(neur_weight >= 0, prev_u, prev_l)

            diffhat = Uhat - Lhat

            value = np.divide(prev_z - Lhat, diffhat)

            indsort = np.argsort(value)

            l_N = np.sum(neur_weight * Lhat) + neur_bias

            if l_N >= 0:
                # TODO: confirm check feasibility: l([n]) > 0?
                aff_w[:,i] = neur_weight
                aff_b[i] = neur_bias

            else:

                l_I = np.sum(neur_weight * Uhat) + neur_bias

                j = -1

                while l_I > 0:
                    j = j + 1
                    l_I = l_I - neur_weight[indsort[j]] * diffhat[indsort[j]]


                h = indsort[j]
                I = indsort[0:(j)]

                l_I = np.sum(neur_weight * Uhat) + neur_bias - np.sum(neur_weight[I] * diffhat[I])

                aff_w[I,i] = neur_weight[I]
                aff_w[h,i] = l_I / diffhat[h]

                aff_b[i] = -(np.sum(neur_weight[I]*Lhat[I]) + l_I*Lhat[h]/diffhat[h])

            v[i] = curr_z[i] - (np.matmul(aff_w[:,i].T, prev_z) + aff_b[i])

        aff_fun = [aff_w, aff_b]

        return aff_fun, v




class AffineFunction:
    # TODO: remove?
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