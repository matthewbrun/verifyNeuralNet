import gurobipy as gp
from gurobipy import GRB


def bigMLP(network):

    # model = gp.Model("test")
    #
    # x = model.addVar(vtype=GRB.BINARY, name="x")
    # y = model.addVar(vtype=GRB.BINARY, name="y")
    # z = model.addVar(vtype=GRB.BINARY, name="z")
    #
    # model.setObjective(x+y+2*z, GRB.MAXIMIZE)
    #
    # model.addConstr(x + 2 * y + 3 * z <= 4, "c0")
    #
    # model.addConstr(x + y >= 1, "c1")
    #
    # model.optimize()
    #
    # print(x.x)
    # print(y.x)
    # print(z.x)
    # print(model.objVal)

