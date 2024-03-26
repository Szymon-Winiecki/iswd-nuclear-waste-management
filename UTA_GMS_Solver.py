import numpy as np
from pulp import *
from matplotlib import pyplot as plt
import os
import networkx as nx

class UTA_GMS_Solver:
    def __init__(self, data, criteria_direction, reference_ranking):
        self.data = data
        self.criteria_direction = criteria_direction
        self.reference_ranking = reference_ranking
        self.num_criterion = self.data.shape[1]

        self.necessary_relation = np.zeros((self.data.shape[0], self.data.shape[0]))
        self.possible_relation = np.zeros((self.data.shape[0], self.data.shape[0]))


    def solve(self):

        for a in range(self.data.shape[0]):
            for b in range(self.data.shape[0]):
                if a == b:
                    continue
                necessary_problem = self.construct_necessary_PL_problem(a, b)
                necessary_problem.solve()
                if LpStatus[necessary_problem.status] == LpStatusInfeasible or value(necessary_problem.objective) <= 0:
                    self.necessary_relation[a, b] = 1
                    self.possible_relation[a, b] = 1
                if self.possible_relation[a, b] == 0:
                    possible_problem = self.construct_possible_PL_problem(a, b)
                    possible_problem.solve()
                    if LpStatus[possible_problem.status] != LpStatusInfeasible and value(possible_problem.objective) > 0:
                        self.possible_relation[a, b] = 1
                    else:
                        self.possible_relation[b, a] = 1


    def construct_base_LP_problem(self, a, b, relation_type):

        problem = LpProblem(f"{relation_type}_relation_{a}>{b}_checking", LpMaximize)

        # create epislon variable
        epsilon = LpVariable("epsilon", lowBound=0, upBound=None)

        # variables for partial utility functions
        utility_variables = []
        # characteristic points x coord
        characteristic_thresholds = []

        for c in range(self.num_criterion):
            characteristic_thresholds.append(np.unique(self.data[:, c]))
            utility_variables.append([LpVariable(f"w_{c}_{i}", lowBound=0, upBound=None) for i in range(len(characteristic_thresholds[c]))])

        # objective function
        problem += epsilon, "Profit"

        # reference ranking

        def utility(c, x):
            threshold = 0
            while x != characteristic_thresholds[c][threshold]:
                threshold += 1
            return utility_variables[c][threshold]


        for e in self.reference_ranking:
            if e[1] == "=":
                problem += lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) == lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)])
            elif e[1] == ">":
                problem += lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)]) + epsilon
            elif e[1] == "<":
                problem += lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) + epsilon

        # normalization
        best_indices = list(map(lambda d : -1 if d == 'gain' else 0, self.criteria_direction))
        worst_indices = list(map(lambda d : 0 if d == 'gain' else -1, self.criteria_direction))
        problem += (lpSum([utility_variables[c][best_indices[c]] for c in range(self.num_criterion)]) == 1, "upper_normalization")
        problem += (lpSum([utility_variables[c][worst_indices[c]] for c in range(self.num_criterion)]) == 0, "lower_normalization")

        # monotonicity
        for c in range(self.num_criterion):
            if self.criteria_direction[c] == 'gain':
                for i in range(1, len(utility_variables[c])):
                    problem += utility_variables[c][i-1] <= utility_variables[c][i]
            else:
                for i in range(1, len(utility_variables[c])):
                    problem += utility_variables[c][i-1] >= utility_variables[c][i]

        return problem, characteristic_thresholds, utility_variables, epsilon
    
    def construct_possible_PL_problem(self, a, b):
        problem, characteristic_thresholds, utility_variables, epsilon = self.construct_base_LP_problem(a, b, 'possible')

        def utility(c, x):
            threshold = 0
            while x != characteristic_thresholds[c][threshold]:
                threshold += 1
            return utility_variables[c][threshold]
        
        problem += lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) + epsilon

        return problem

    def construct_necessary_PL_problem(self, a, b):
        problem, characteristic_thresholds, utility_variables, epsilon = self.construct_base_LP_problem(a, b, 'necessary')

        def utility(c, x):
            threshold = 0
            while x != characteristic_thresholds[c][threshold]:
                threshold += 1
            return utility_variables[c][threshold]
        
        problem += lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) + epsilon

        return problem


    def hasse_diagram(self):
        G = nx.DiGraph()
        
        for a in range(self.necessary_relation.shape[0]):
            for b in range(self.necessary_relation.shape[1]):
                if self.necessary_relation[a, b] == 1:
                    if not G.has_node(a+1):
                        G.add_node(a+1)
                    if not G.has_node(b+1):
                        G.add_node(b+1)
                    G.add_edge(a+1, b+1)

        edges = [e for e in G.edges]
        for a, b in edges:
            G.remove_edge(a, b)
            if not nx.has_path(G, a, b):
                G.add_edge(a, b)
        
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()