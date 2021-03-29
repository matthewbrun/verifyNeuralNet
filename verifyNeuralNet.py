import gurobipy as gp
from gurobipy import GRB

import numpy as np

import nnet


def bigMLP(network, input, inf_d, real_class, pred_class, method):
    """
    Solves an LP relaxation of the big M formulation of the ReLU activations in provided network to maximize the
    difference between desired pred_class and actual real_class of the input within an infinity norm distance of inf_d

    :param network: keras neural network object
    :param input: numeric input around witch to find adversarial example
    :param inf_d: infinity norm distance around input to search for example
    :param real_class: output class of input
    :param pred_class: desired class of adversarial example
    :param method: method for bound propogation, 1 = interval arithmetic
    :return: adversarial example, objective value
    """

    #Create nnet object and generate numeric bounds
    n = nnet.Sequential(network)
    n.generate_bounds(method, input, inf_d)

    #Create model
    m = gp.Model("bigM")

    #Add bounded variable on the input space
    input_lb = input - inf_d
    input_ub = input + inf_d
    x = m.addMVar(shape = n.input_shape, lb = input_lb, ub = input_ub, name = "input")

    layer_vars = {"input":x} #Dictionary of layer output variables

    for i, layer in enumerate(n.layers): #Iterate over each layer

        #Get name for layer and previous layer
        name = layer.label
        if i > 0:
            prev_layer = n.layers[i-1]
            prev_name = prev_layer.label
        else:
            prev_name = "input"


        w = layer.weights
        b = layer.bias
        Mc = np.divide(layer.aff_ub, (layer.aff_ub - layer.aff_lb))

        x = layer_vars[prev_name] #Output variable from previous layer

        #Add output variable for layer
        y = m.addMVar(shape = layer.output_shape, lb = 0 ,name = name)

        for j in range(layer.output_shape): #For each neuron
            if layer.aff_ub[j] <= 0: #if always inactive, set output to 0
                m.addConstr(y[j] == 0, "zero" + str(i)+"."+str(j))

            elif layer.aff_lb[j] >= 0: #if always active, set output to the affine function
                m.addConstr(y[j] == w.T[j, :] @ x + b[j], "aff" + str(i)+"."+str(j))

            else: #otherwise, add bigM ReLU constraints
                m.addConstr(y[j] >= w.T[j,:] @ x + b[j], "lb" + str(i)+"."+str(j))
                m.addConstr(y[j] <= Mc[j] * ((w.T[j, :] @ x) + b[j] - layer.aff_lb[j]), "ub" + str(i)+"."+str(j))

        layer_vars[name] = y

    y = layer_vars[n.layers[-1].label] #output variables
    x = layer_vars["input"] #input variables

    m.setObjective(y[pred_class] - y[real_class], GRB.MAXIMIZE) #Maximize difference between desired output class and actual class

    m.setParam('OutputFlag', 0)
    m.optimize()

    return x.X, m.objVal

