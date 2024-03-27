import numpy as np
from pulp import *
from matplotlib import pyplot as plt
import os

class UTA_REP_Solver:
    def __init__(self, data, criteria_direction, reference_ranking, necessary_relation):
        self.data = data
        self.criteria_direction = criteria_direction
        self.reference_ranking = reference_ranking
        self.num_criterion = self.data.shape[1]

        self.necessary_relation = necessary_relation


    def solve(self):

        self.construct_LP_problem()

        self.problem.solve()

        self.characteristic_values = []
        for c in range(len(self.utility_variables)):
            self.characteristic_values.append([])
            for var in self.utility_variables[c]:
                self.characteristic_values[c].append(var.varValue)


    def construct_LP_problem(self):

        self.problem = LpProblem(f"find_representative_function", LpMaximize)

        # create epislon variable
        epsilon = LpVariable("epsilon", lowBound=0, upBound=None)

        delta = LpVariable("delta", lowBound=0, upBound=None)

        # variables for partial utility functions
        self.utility_variables = []
        # characteristic points x coord
        self.characteristic_thresholds = []

        for c in range(self.num_criterion):
            self.characteristic_thresholds.append(np.unique(self.data[:, c]))
            self.utility_variables.append([LpVariable(f"w_{c}_{i}", lowBound=0, upBound=None) for i in range(len(self.characteristic_thresholds[c]))])

        M = 1000

        # objective function
        self.problem += M * epsilon - delta, "Profit"

        # reference ranking

        def utility(c, x):
            threshold = 0
            while x != self.characteristic_thresholds[c][threshold]:
                threshold += 1
            return self.utility_variables[c][threshold]


        for e in self.reference_ranking:
            if e[1] == "=":
                self.problem += lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) == lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)])
            elif e[1] == ">":
                self.problem += lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)]) + epsilon
            elif e[1] == "<":
                self.problem += lpSum([utility(c, self.data[e[2], c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[e[0], c]) for c in range(self.num_criterion)]) + epsilon

        # normalization
        best_indices = list(map(lambda d : -1 if d == 'gain' else 0, self.criteria_direction))
        worst_indices = list(map(lambda d : 0 if d == 'gain' else -1, self.criteria_direction))
        self.problem += (lpSum([self.utility_variables[c][best_indices[c]] for c in range(self.num_criterion)]) == 1, "upper_normalization")
        self.problem += (lpSum([self.utility_variables[c][worst_indices[c]] for c in range(self.num_criterion)]) == 0, "lower_normalization")

        # monotonicity
        for c in range(self.num_criterion):
            if self.criteria_direction[c] == 'gain':
                for i in range(1, len(self.utility_variables[c])):
                    self.problem += self.utility_variables[c][i-1] <= self.utility_variables[c][i]
            else:
                for i in range(1, len(self.utility_variables[c])):
                    self.problem += self.utility_variables[c][i-1] >= self.utility_variables[c][i]

        # rep
        for a in range(self.necessary_relation.shape[0]):
            for b in range(a):
                if self.necessary_relation[a, b] == 1:
                    self.problem += lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) + epsilon
                elif self.necessary_relation[b, a] == 1:
                    self.problem += lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) >= lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) + epsilon
                else:
                    self.problem += lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) - lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) <= delta
                    self.problem += lpSum([utility(c, self.data[b, c]) for c in range(self.num_criterion)]) - lpSum([utility(c, self.data[a, c]) for c in range(self.num_criterion)]) <= delta


    def rank(self):
        ranking = []
        for i in range(self.data.shape[0]):
            ranking.append((i, self.utility(self.data[i])))

        ranking.sort(key = lambda x : -x[1])

        return ranking

    def partial_utility(self, x, characteristic_thresholds, characteristic_values):
        threshold = 0
        while x != characteristic_thresholds[threshold]:
            threshold += 1
        return characteristic_values[threshold]

    def utility(self, variant):
        u = 0
        for i in range(len(variant)):
            u += self.partial_utility(variant[i], self.characteristic_thresholds[i],  self.characteristic_values[i])
        return u

    def plot_partial_utility(self, directory="results/plots_UTA_REP"):

        os.makedirs(directory, exist_ok=True)

        for c in range(len(self.characteristic_thresholds)):
            plt.cla()
            plt.ylim(0.0, 1.0)
            plt.plot(self.characteristic_thresholds[c], self.characteristic_values[c], marker="o")
            plt.savefig(os.path.join(directory, f"U_{c}.png"))