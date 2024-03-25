import numpy as np
from pulp import *
from matplotlib import pyplot as plt
import os

class UTA_Solver:
    def __init__(self, data, criteria_direction, reference_ranking, num_characteristic_points, utility_min_weight = 0.0, utility_max_weight = 1.0):
        self.data = data
        self.criteria_direction = criteria_direction
        self.reference_ranking = reference_ranking
        self.num_characteristic_points = num_characteristic_points
        self.num_criterion = self.data.shape[1]
        self.utility_min_weight = utility_min_weight
        self.utility_max_weight = utility_max_weight


    def solve(self):

        self.construct_LP_problem()

        self.problem.solve()

        self.characteristic_values = []
        for c in range(len(self.utility_variables)):
            self.characteristic_values.append([])
            for var in self.utility_variables[c]:
                self.characteristic_values[c].append(var.varValue)

    def construct_LP_problem(self):

        self.problem = LpProblem("Maximize_the_epsilon", LpMaximize)

        # create epislon variable
        epsilon = LpVariable("epsilon", lowBound=0, upBound=None)

        # variables for partial utility functions
        self.utility_variables = []
        # characteristic points x coord
        self.characteristic_thresholds = []

        for c in range(self.num_criterion):
            self.utility_variables.append([])
            for i in range(self.num_characteristic_points):
                self.utility_variables[c].append(LpVariable(f"w_{c}_{i}", lowBound=0, upBound=None))
            self.characteristic_thresholds.append(np.linspace(self.data[:,c].min(), self.data[:,c].max(), self.num_characteristic_points))

        # objective function
        self.problem += epsilon, "Profit"

        # reference ranking

        def utility(c, x):
            succ_theshold = 1
            while x > self.characteristic_thresholds[c][succ_theshold]:
                succ_theshold += 1
            return self.utility_variables[c][succ_theshold - 1] + ( (x - self.characteristic_thresholds[c][succ_theshold - 1]) / (self.characteristic_thresholds[c][succ_theshold] - self.characteristic_thresholds[c][succ_theshold - 1]) ) * (self.utility_variables[c][succ_theshold] - self.utility_variables[c][succ_theshold - 1])


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

        # min and max criteria weights
        for c in range(self.num_criterion):
            self.problem += self.utility_variables[c][best_indices[c]] >= self.utility_min_weight
            self.problem += self.utility_variables[c][best_indices[c]] <= self.utility_max_weight

        # monotonicity
        for c in range(self.num_criterion):
            if self.criteria_direction[c] == 'gain':
                for i in range(1, len(self.utility_variables[c])):
                    self.problem += self.utility_variables[c][i-1] <= self.utility_variables[c][i]
            else:
                for i in range(1, len(self.utility_variables[c])):
                    self.problem += self.utility_variables[c][i-1] >= self.utility_variables[c][i]

    def rank(self):
        ranking = []
        for i in range(self.data.shape[0]):
            ranking.append((i, self.utility(self.data[i])))

        ranking.sort(key = lambda x : -x[1])

        return ranking

    def partial_utility(self, x, characteristic_thresholds, characteristic_values):
        succ_theshold = 1
        while x > characteristic_thresholds[succ_theshold]:
            succ_theshold += 1
        return characteristic_values[succ_theshold - 1] + ( (x - characteristic_thresholds[succ_theshold - 1]) / (characteristic_thresholds[succ_theshold] - characteristic_thresholds[succ_theshold - 1]) ) * (characteristic_values[succ_theshold] - characteristic_values[succ_theshold - 1])

    def utility(self, variant):
        u = 0
        for i in range(len(variant)):
            u += self.partial_utility(variant[i], self.characteristic_thresholds[i],  self.characteristic_values[i])
        return u

    def plot_partial_utility(self, directory="results/plots"):

        os.makedirs(directory, exist_ok=True)

        for c in range(len(self.characteristic_thresholds)):
            plt.cla()
            plt.plot(self.characteristic_thresholds[c], self.characteristic_values[c])
            plt.savefig(os.path.join(directory, f"U_{c}.png"))