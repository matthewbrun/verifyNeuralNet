import numpy as np
import keras

import gurobipy as gp
from gurobipy import GRB

#TODO: reduce use of np.copy - only necessary when resulting object edited
#TODO: dynamic error sizes
#TODO: add tolerances to tighten Gurobi constraints
#TODO: add tolerances to make neuron (in)activity more lenient

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

    def generate_bounds(self, input, distance, options = False):
        """
        Generate upper and lower bounds on the affine function within each neuron

        :param input: numeric input to verify
        :param distance: l_inf distance around input to consider
        :param options: BoundsOptions object specifying method and options,
                methods include: IntervalArithmetic, DeepPoly, FastC2V, FlatC2V
        """

        #Get options
        if not options:
            options = BoundsOptions('IntervalArithmetic')
        method = options.method

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

        elif method == 2: #DeepPoly

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
                if i > 0:
                    layer.generate_DeepPoly_bounds(np.copy(self.layers[i-1].numeric_relu_lb), np.copy(self.layers[i-1].numeric_relu_ub))
                else:
                    layer.generate_DeepPoly_bounds(input_lb, input_ub)

                aff_lbs.append( [np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)] )
                aff_ubs.append( [np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)] )

        elif method == 3: #FastC2V

            #Keep affine bounding functions from previous layers in the form (weights, biases)
            aff_lbs = []
            aff_ubs = []


            for i, layer in enumerate(self.layers):

                #Generate lower and upper bounds and solution data from a backwards pass
                L, U, x_lb, x_ub, top_lbs, top_ubs = self.backwards_pass(i, aff_lbs, aff_ubs, input_lb, input_ub)

                #Find solution from backwards pass data
                z_lb, z_ub = self.forwards_pass(i, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub, options.use_FP_relu)


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

                        #Take ReLU of inputs if specified in parameter
                        if k > 0 and options.use_FP_relu:
                            z_lb_inp = np.maximum(z_lb[k][:,j],0)
                            z_ub_inp = np.maximum(z_ub[k][:,j], 0)
                        else:
                            z_lb_inp = z_lb[k][:, j]
                            z_ub_inp = z_ub[k][:, j]

                        new_aff_lb, viol_lb = tighten_layer.most_violated_inequality(prev_l, prev_u, z_lb_inp, z_lb[k+1][:,j]) #lb

                        new_aff_ub, viol_ub = tighten_layer.most_violated_inequality(prev_l, prev_u, z_ub_inp, z_ub[k+1][:,j]) #ub

                        #If the inequalities are violated, replace for this neuron
                        new_aff_ubs_lb[k][0] = np.where(viol_lb > 0, new_aff_lb[0], new_aff_ubs_lb[k][0])
                        new_aff_ubs_ub[k][0] = np.where(viol_ub > 0, new_aff_ub[0], new_aff_ubs_ub[k][0])

                        new_aff_ubs_lb[k][1] = np.where(viol_lb > 0, new_aff_lb[1], new_aff_ubs_lb[k][1])
                        new_aff_ubs_ub[k][1] = np.where(viol_ub > 0, new_aff_ub[1], new_aff_ubs_ub[k][1])

                        #Update numeric bounds for next layer
                        prev_l = np.copy(tighten_layer.numeric_relu_lb)
                        prev_u = np.copy(tighten_layer.numeric_relu_ub)

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
                if i > 0:
                    layer.generate_DeepPoly_bounds(np.copy(self.layers[i-1].numeric_relu_lb), np.copy(self.layers[i-1].numeric_relu_ub))
                else:
                    layer.generate_DeepPoly_bounds(input_lb, input_ub)

                aff_lbs.append( [np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)] )
                aff_ubs.append( [np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)] )

        elif method == 4: #FlatC2V

            # Keep affine bounding functions from previous layers in the form (weights, biases)
            aff_lbs = []
            aff_ubs = []

            flat_aff_lbs = [] #TODO: add lb options?
            flat_aff_ubs = []


            for i, layer in enumerate(self.layers):

                if i > 0:
                    flat_aff_ubs.append( layer.flattest_inequality(np.copy(self.layers[i - 1].numeric_relu_lb),
                                                   np.copy(self.layers[i - 1].numeric_relu_ub)) )
                else:
                    flat_aff_ubs.append( layer.flattest_inequality(input_lb, input_ub) )

                # Generate lower and upper bounds and solution data from a backwards pass
                L, U, x_lb, x_ub, top_lbs, top_ubs = self.backwards_pass(i, aff_lbs, aff_ubs, input_lb, input_ub)

                if options.do_iterative_tighten:

                    # Find solution from backwards pass data
                    z_lb, z_ub = self.forwards_pass(i, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub, options.use_FP_relu)

                    # This is inefficient: loops over each neuron in layer
                    # is it possible to vectorize these computations?

                    # Generate new tightened numeric bounds at each neuron in present layer
                    for j in range(layer.output_shape):
                        new_aff_ubs_lb = []
                        new_aff_ubs_ub = []
                        for item in flat_aff_ubs:
                            new_aff_ubs_lb.append(
                                [np.copy(item[0]), np.copy(item[1])])  # replaced affine upper bounds for neuron lower bound
                            new_aff_ubs_ub.append(
                                [np.copy(item[0]), np.copy(item[1])])  # replaced affine upper bounds for neuron upper bound

                        if options.use_viol:
                            # Tighten bounds on each preceeding layer
                            for k, tighten_layer in enumerate(self.layers[0:i]):

                                # Find most violated inequalities given solution for fixed neuron and lower/upper bounded solutions

                                # Take ReLU of inputs if specified in parameter
                                if k > 0 and options.use_FP_relu:
                                    z_lb_inp = np.maximum(z_lb[k][:, j], 0)
                                    z_ub_inp = np.maximum(z_ub[k][:, j], 0)
                                else:
                                    z_lb_inp = z_lb[k][:, j]
                                    z_ub_inp = z_ub[k][:, j]

                                viol_lb = z_lb[k + 1][:, j] - (np.matmul(tighten_layer.weights.T, z_lb_inp) + tighten_layer.bias)  # lb
                                viol_ub = z_ub[k + 1][:, j] - (np.matmul(tighten_layer.weights.T, z_ub_inp) + tighten_layer.bias) # ub

                                # If the inequalities are violated, replace for this neuron
                                new_aff_ubs_lb[k][0] = np.where(viol_lb > 0, flat_aff_ubs[k][0], new_aff_ubs_lb[k][0])
                                new_aff_ubs_ub[k][0] = np.where(viol_ub > 0, flat_aff_ubs[k][0], new_aff_ubs_ub[k][0])

                                new_aff_ubs_lb[k][1] = np.where(viol_lb > 0, flat_aff_ubs[k][1], new_aff_ubs_lb[k][1])
                                new_aff_ubs_ub[k][1] = np.where(viol_ub > 0, flat_aff_ubs[k][1], new_aff_ubs_ub[k][1])

                        # This is inefficient: only need backwards pass to a single neuron in the last layer, not all neurons
                        # Also, only need lower or upper bound at a time, not both

                        # Repeat backwards pass with new affine bounds for current neuron
                        Lj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_lb, input_lb, input_ub)
                        __, Uj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_ub, input_lb, input_ub)

                        # Replace numeric bounds at current neuron if improved
                        L[j] = max(Lj_new[j], L[j])
                        U[j] = min(Uj_new[j], U[j])

                # Set layer numeric lower and upper affine bounds
                layer.numeric_aff_ub = U
                layer.numeric_aff_lb = L

                # Generate lower and upper bounding affine function for layer
                if i > 0:
                    layer.generate_DeepPoly_bounds(np.copy(self.layers[i - 1].numeric_relu_lb),
                                                   np.copy(self.layers[i - 1].numeric_relu_ub))
                else:
                    layer.generate_DeepPoly_bounds(input_lb, input_ub)

                aff_lbs.append([np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)])

                if options.use_flat_ubs:
                    aff_ubs.append([np.copy(flat_aff_ubs[i][0]), np.copy(flat_aff_ubs[i][1])])
                    layer.funcw_aff_ub = flat_aff_ubs[i][0]
                    layer.funcb_aff_ub = flat_aff_ubs[i][1]
                else:
                    aff_ubs.append([np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)])

        elif method == 5: #MeanC2V

            points = 2 * distance * np.random.rand(len(input), options.num_points) - distance + np.tile(input, (options.num_points, 1)).T
            mean_pts = [np.mean(points, axis=1)]

            # Keep affine bounding functions from previous layers in the form (weights, biases)
            aff_lbs = []
            aff_ubs = []

            mean_aff_lbs = [] #TODO: add lb options?
            mean_aff_ubs = []

            for i, layer in enumerate(self.layers):

                # Generate lower and upper bounds and solution data from a backwards pass
                L, U, x_lb, x_ub, top_lbs, top_ubs = self.backwards_pass(i, aff_lbs, aff_ubs, input_lb, input_ub)

                # Find solution from backwards pass data
                z_lb, z_ub = self.forwards_pass(i, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub, options.use_FP_relu)

                #Compute propogation of points
                points = layer.evaluate(points, options.use_FP_relu)
                mean_pts.append(np.mean(points, axis=1))

                if i > 0:
                    mean_aff_ubs.append(layer.most_violated_inequality(np.copy(self.layers[i - 1].numeric_relu_lb),
                                                   np.copy(self.layers[i - 1].numeric_relu_ub), mean_pts[i], layer.evaluate(mean_pts[i], options.use_FP_relu))[0])
                else:
                    mean_aff_ubs.append(layer.most_violated_inequality(input_lb, input_ub, mean_pts[i], layer.evaluate(mean_pts[i], options.use_FP_relu))[0])

                # This is inefficient: loops over each neuron in layer
                # is it possible to vectorize these computations?

                # Generate new tightened numeric bounds at each neuron in present layer
                for j in range(layer.output_shape):
                    new_aff_ubs_lb = []
                    new_aff_ubs_ub = []
                    for item in aff_ubs:
                        new_aff_ubs_lb.append(
                            [np.copy(item[0]), np.copy(item[1])])  # replaced affine upper bounds for neuron lower bound
                        new_aff_ubs_ub.append(
                            [np.copy(item[0]), np.copy(item[1])])  # replaced affine upper bounds for neuron upper bound

                    if options.use_viol:
                        # Tighten bounds on each preceeding layer
                        for k, tighten_layer in enumerate(self.layers[0:i]):

                            # Find most violated inequalities given solution for fixed neuron and lower/upper bounded solutions

                            # Take ReLU of inputs if specified in parameter
                            if k > 0 and options.use_FP_relu:
                                z_lb_inp = np.maximum(z_lb[k][:, j], 0)
                                z_ub_inp = np.maximum(z_ub[k][:, j], 0)
                            else:
                                z_lb_inp = z_lb[k][:, j]
                                z_ub_inp = z_ub[k][:, j]

                            viol_lb = z_lb[k + 1][:, j] - (np.matmul(tighten_layer.weights.T, z_lb_inp) + tighten_layer.bias)  # lb
                            viol_ub = z_ub[k + 1][:, j] - (np.matmul(tighten_layer.weights.T, z_ub_inp) + tighten_layer.bias)  # ub

                            # If the inequalities are violated, replace for this neuron
                            new_aff_ubs_lb[k][0] = np.where(viol_lb > 0, mean_aff_ubs[k][0], new_aff_ubs_lb[k][0])
                            new_aff_ubs_ub[k][0] = np.where(viol_ub > 0, mean_aff_ubs[k][0], new_aff_ubs_ub[k][0])

                            new_aff_ubs_lb[k][1] = np.where(viol_lb > 0, mean_aff_ubs[k][1], new_aff_ubs_lb[k][1])
                            new_aff_ubs_ub[k][1] = np.where(viol_ub > 0, mean_aff_ubs[k][1], new_aff_ubs_ub[k][1])

                    # This is inefficient: only need backwards pass to a single neuron in the last layer, not all neurons
                    # Also, only need lower or upper bound at a time, not both

                    # Repeat backwards pass with new affine bounds for current neuron
                    Lj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_lb, input_lb, input_ub)
                    __, Uj_new, *__ = self.backwards_pass(i, aff_lbs, new_aff_ubs_ub, input_lb, input_ub)

                    # Replace numeric bounds at current neuron if improved
                    L[j] = max(Lj_new[j], L[j])
                    U[j] = min(Uj_new[j], U[j])

                # Set layer numeric lower and upper affine bounds
                layer.numeric_aff_ub = U
                layer.numeric_aff_lb = L

                # Generate lower and upper bounding affine function for layer
                if i > 0:
                    layer.generate_DeepPoly_bounds(np.copy(self.layers[i - 1].numeric_relu_lb),
                                                   np.copy(self.layers[i - 1].numeric_relu_ub))
                else:
                    layer.generate_DeepPoly_bounds(input_lb, input_ub)

                aff_lbs.append([np.copy(layer.funcw_aff_lb), np.copy(layer.funcb_aff_lb)])

                if options.use_mean_ubs:
                    aff_ubs.append([np.copy(mean_aff_ubs[i][0]), np.copy(mean_aff_ubs[i][1])])
                    layer.funcw_aff_ub = mean_aff_ubs[i][0]
                    layer.funcb_aff_ub = mean_aff_ubs[i][1]
                else:
                    aff_ubs.append([np.copy(layer.funcw_aff_ub), np.copy(layer.funcb_aff_ub)])


            return points



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



    def forwards_pass(self, l_num, aff_lbs, aff_ubs, top_lbs, top_ubs, x_lb, x_ub, use_relu = True):
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
                        (np.matmul(aff_ubs[i][0].T, prev_ub) + np.tile(np.reshape(aff_ubs[i][1], (-1,1)), (1,n_neur))) * top_ubs[i]


            #Add layer solution to z vector
            z_lb.append(next_z_lb)
            z_ub.append(next_z_ub)

            #Pass outputs through ReLU if specified in parameter
            if use_relu:
                prev_lb = np.copy(np.maximum(next_z_lb, 0))
                prev_ub = np.copy(np.maximum(next_z_ub, 0))
            else:
                prev_lb = next_z_lb
                prev_ub = next_z_ub


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

    def evaluate(self, points, use_relu = True):
        """
        Evaluate a set of input points through the neuron
        :param points: points at which to evaluate the neuron, each point in a column
        :param use_relu: boolean indicator of whether to take ReLU transformation of outputs
        :return: out: output values
        """

        #TODO: remove cases?
        num_pts = 1
        s = points.shape
        if len(s) > 1:
            out = np.matmul(self.weights.T, points) + np.tile(self.bias, (num_pts, 1)).T
            num_pts = s[1]

        else:
            out = np.matmul(self.weights.T, points) + self.bias

        if use_relu:
            out = np.maximum(out, 0)

        return out

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

    def generate_DeepPoly_bounds(self, prev_l, prev_u):
        """
        Generate affine bounding functions on neuron layer using the DeepPoly method
        Requires self.numeric_aff_lb and self.numeric_aff_ub be set to numeric upper and lower bounds on each neruon
        :param prev_l: lower bound on previous layer
        :param prev_u: upper bound on previous layer
        """

        L = np.copy(self.numeric_aff_lb)
        U = np.copy(self.numeric_aff_ub)

        #Take better of interval arithmetic and provided bounds
        self.generate_interval_bounds(prev_l, prev_u)

        L = np.maximum(np.copy(self.numeric_aff_lb), L)
        U = np.minimum(np.copy(self.numeric_aff_ub), U)
        self.numeric_aff_lb = L
        self.numeric_aff_ub = U

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

    def most_violated_inequality(self, prev_l, prev_u, prev_z, curr_z):
        """
        Finds the affine representation of the most violated inequality from the convex relaxation for a neuron
        :param prev_l: numeric lower bounds on the previous layer
        :param prev_u: numeric upper bounds on the previous layer
        :param prev_z: forward pass solution in the previous layer for neuron
        :param curr_z: output of neuron
        :return: aff_fun, v: affine inequality function (weights, bias), violation
        """

        #Get current upper bounding function
        aff_w = np.zeros(self.weights.shape)
        aff_b = np.zeros(self.bias.shape)

        #Violations at solution for each neuron
        v = np.zeros((self.output_shape))

        #Find inequality at each neuron
        for i in range(self.output_shape):

            #Get neuron affine function
            neur_weight = np.copy(self.weights[:,i])
            neur_bias = np.copy(self.bias[i])

            #Construct lower/upper bound objects
            Lhat = np.where(neur_weight >= 0, prev_l, prev_u)
            Uhat = np.where(neur_weight >= 0, prev_u, prev_l)

            diffhat = Uhat - Lhat

            #Find values for knapsack algorithm
            value = np.full(diffhat.shape, float('inf'))
            value = np.divide(prev_z - Lhat, diffhat, out = value, where = diffhat != 0)

            indsort = np.argsort(value) #indices in order of smallest value

            l_N = np.sum(neur_weight * Lhat) + neur_bias #l([n])
            l_0 = np.sum(neur_weight * Uhat) + neur_bias #l(0)

            #Check feasibility
            if l_N >= 0:
                #Neuron always active
                aff_w[:,i] = np.copy(self.weights[:,i])
                aff_b[i] = np.copy(self.bias[i])

                assert (abs(curr_z[i] - (np.matmul(aff_w[:,i].T, prev_z) + aff_b[i])) < 1e-10)

            elif l_0 < 0:
                #Neuron always inactive

                assert (abs(curr_z[i] - (np.matmul(aff_w[:, i].T, prev_z) + aff_b[i])) < 1e-10)

                pass

            else:

                l_I = l_0 #l(0)

                #Add indices to I/h until condition on l is met
                j = -1

                while l_I >= 0:
                    j = j + 1
                    l_I = l_I - neur_weight[indsort[j]] * diffhat[indsort[j]]

                #Collect indices selected
                h = indsort[j]
                I = indsort[0:(j)]

                #Compute final value of l(I)
                l_I = np.sum(neur_weight * Uhat) + neur_bias - np.sum(neur_weight[I] * diffhat[I])

                assert (l_I >= -(1e-10))
                assert (l_I - neur_weight[h]*diffhat[h] < 1e-10)

                #Replace affine inequality with that represented by I/h
                aff_w[I,i] = neur_weight[I]
                aff_w[h,i] = l_I / diffhat[h]

                aff_b[i] = -(np.sum(neur_weight[I]*Lhat[I]) + l_I*Lhat[h]/diffhat[h])

            #Compute violation of inequality at solution
            v[i] = curr_z[i] - (np.matmul(aff_w[:,i].T, prev_z) + aff_b[i])

        aff_fun = [aff_w, aff_b]

        return aff_fun, v

    def flattest_inequality(self, prev_l, prev_u):
        """
        Finds the affine representation of the flattest inequality from the convex relaxation for a neuron
        :param prev_l: numeric lower bounds on the previous layer
        :param prev_u: numeric upper bounds on the previous layer
        :return: aff_fun: affine inequality function (weights, bias)
        """

        #Get current upper bounding function
        aff_w = np.zeros(self.weights.shape)
        aff_b = np.zeros(self.bias.shape)

        #Find inequality at each neuron
        for i in range(self.output_shape):

            #Get neuron affine function
            neur_weight = np.copy(self.weights[:,i])
            neur_bias = np.copy(self.bias[i])

            #Construct lower/upper bound objects
            Lhat = np.where(neur_weight >= 0, prev_l, prev_u)
            Uhat = np.where(neur_weight >= 0, prev_u, prev_l)

            diffhat = Uhat - Lhat

            l_N = np.sum(neur_weight * Lhat) + neur_bias #l([n])
            l_0 = np.sum(neur_weight * Uhat) + neur_bias #l(0)

            #Check feasibility
            if l_N >= 0:
                #Neuron always active
                aff_w[:,i] = np.copy(self.weights[:,i])
                aff_b[i] = np.copy(self.bias[i])

                #assert (abs(curr_z[i] - (np.matmul(aff_w[:,i].T, prev_z) + aff_b[i])) < 1e-10)

            elif l_0 < 0:
                #Neuron always inactive

                #assert (abs(curr_z[i] - (np.matmul(aff_w[:, i].T, prev_z) + aff_b[i])) < 1e-10)

                pass

            else:

                ######FlatCut1 Model Here#######

                #Set initial I,h,cost solution
                c = float('inf')
                h = 0
                I = []

                #Compute set of potential optimal h based on largest w*(U-L)
                value = neur_weight*diffhat
                indsort = np.argsort(value)  # indices in order of smallest value

                hset = [] #Set of candidate h
                l_hset = l_0

                h_idx = self.input_shape - 1
                while l_hset >= 0: #l(hset) must be negative to have all h)

                    #Add next largest h to hset
                    hnext = indsort[h_idx]
                    hset.append(hnext)

                    l_hset = l_hset - value[hnext]
                    h_idx = h_idx - 1

                #Solve QKP over all candidate h, keep solution with lowest cost
                for hc in hset:

                    mpt = np.sum(neur_weight*(Uhat+Lhat)) + 2*neur_bias

                    m = gp.Model("FlatCut1")

                    u = m.addMVar(shape = (self.input_shape), vtype = GRB.BINARY) #binary inclusion variables for I

                    #Add necessary constraint
                    if mpt < 0:
                        m.addConstr((neur_weight*diffhat) @ u <= np.sum(neur_weight*Uhat) + neur_bias, "f(Il)")
                    else:
                        m.addConstr((-neur_weight*diffhat) @ u <= neur_weight[hc]*(diffhat[hc]) - np.sum(neur_weight*Uhat) - neur_bias, "f(Ir)")

                    #Fix h not in I
                    m.addConstr(u[hc] == 0, "[n]\h")

                    #Build objective function
                    qMat = np.outer(neur_weight*diffhat,neur_weight*diffhat)/(neur_weight[hc]*diffhat[hc]) #quadratic term matrix
                    obj = (u @ qMat @ u) + neur_weight*diffhat*(1-(np.sum(neur_weight*diffhat))/(neur_weight[hc]*diffhat[hc])) @ u

                    m.setObjective(obj, GRB.MINIMIZE)
                    m.setParam('OutputFlag', 0)

                    #TODO: what should this tolerance be
                    m.setParam('FeasibilityTol', 1e-9)
                    m.setParam('OptimalityTol', 1e-9)

                    m.optimize()

                    #Extract optimal candidate I
                    Ic = np.where(u.X > .5)
                    l_Ic = np.sum(neur_weight * Uhat) + neur_bias - np.sum(neur_weight[Ic] * diffhat[Ic])

                    #Confirm validity of computed solution
                    assert (l_Ic >= -1e-5)
                    assert (l_Ic - neur_weight[hc] * diffhat[hc] < 1e-5)

                    #Compute objective cost of candidate I/h pair
                    cc = (l_Ic / (neur_weight[hc]*diffhat[hc]) - 1)*(l_Ic - mpt)

                    #If cost improved, keep current solution
                    if cc < c:
                        I = Ic
                        h = hc
                        c = cc

                ######End Model######

                #Keep I and h, objective value

                #Compute final value of l(I)
                l_I = np.sum(neur_weight * Uhat) + neur_bias - np.sum(neur_weight[I] * diffhat[I])

                assert (l_I >= -1e-5)
                assert (l_I - neur_weight[h]*diffhat[h] < 1e-5)

                #Replace affine inequality with that represented by I/h
                aff_w[I,i] = neur_weight[I]
                aff_w[h,i] = l_I / diffhat[h]

                aff_b[i] = -(np.sum(neur_weight[I]*Lhat[I]) + l_I*Lhat[h]/diffhat[h])

        aff_fun = [aff_w, aff_b]

        return aff_fun


class BoundsOptions():

    def __init__(self, method, use_FP_relu = True, use_viol = False, do_iterative_tighten = True,
                        use_flat_ubs = False, use_mean_ubs = False, num_points = 100):
        """
        :param method: method for bound propogation, 1 = interval arithmetic, 2 = DeepPoly, 3 = FastC2V, 4 = FlatC2V
        :param use_FP_relu: boolean indicator of whether to use ReLU tightening between layers of forwards pass
        :param use_viol: boolean indicator of whether to require a cut be violated to replace (FlatC2V only)
        :param do_iterative_tighten: boolean indicator of whether to tighten a neuron with new bounds from the previous
        :param use_flat_ubs: boolean indicator of whether to use affine inequalities from flat cuts instead of DeepPoly
        """

        valid_methods = ["IntervalArithmetic", "DeepPoly", "FastC2V", "FlatC2V", "MeanC2V"]
        if method not in valid_methods:
            raise(Exception("Invalid method.  Valid methods include: " + str(valid_methods)))

        self.method = valid_methods.index(method) + 1

        if self.method == 3: #FastC2V
            self.use_FP_relu = use_FP_relu

        elif self.method == 4: #FlatC2V
            self.use_FP_relu = use_FP_relu
            self.use_viol = use_viol
            self.do_iterative_tighten = do_iterative_tighten
            self.use_flat_ubs = use_flat_ubs

            if use_viol and not do_iterative_tighten:
                raise(Exception("Must do iterative tightening (do_iterative_tighten) to check violations (use_viol)"))

        elif self.method == 5: #MeanC2V
            self.num_points = num_points
            self.use_FP_relu = use_FP_relu
            self.use_viol = use_viol
            self.use_mean_ubs = use_mean_ubs




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