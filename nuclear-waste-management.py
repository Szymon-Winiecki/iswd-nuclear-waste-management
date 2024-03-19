import csv
import numpy as np
import sys
from pulp import *
from matplotlib import pyplot as plt
import os

def main():
    data_file = 'Nuclear waste management.csv'
    data = read_data(data_file)

    criteria_direction = ['cost', 'cost', 'cost', 'cost']

    reference_ranking = [
        (11, '>', 18),
        (7,  '>', 21),
        (16, '>', 3),
        (13, '>', 26),
        (2,  '>', 8)
    ]

    num_characteristic_points = 3

    construct_LP_problem(data, criteria_direction, reference_ranking, num_characteristic_points)


def read_data(filename):
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        first_row = True
        for row in reader:
            if first_row:
                first_row = False
                continue
            data.append(row[1:])

    return np.array(data, dtype=float)

def construct_LP_problem(data, criteria_direction, reference_ranking, num_characteristic_points):

    best_values = []
    worst_values = []
    for c in range(data.shape[1]):
        if criteria_direction[c] == 'gain':
            best_values.append(data[:,c].max())
            worst_values.append(data[:,c].min())
        elif criteria_direction[c] == 'cost':
            best_values.append(data[:,c].min())
            worst_values.append(data[:,c].max())
        else:
            sys.exit(f"unknown direction: {criteria_direction[c]}")

    prob = LpProblem("Maximize the epsilon", LpMaximize)

    # create epislon variable
    epsilon = LpVariable("epsilon", lowBound=0, upBound=None)

    # variables for partial utility functions
    utility_variables = []
    # characteristic points x coord
    utility_thresholds = []

    for c in range(data.shape[1]):
        utility_variables.append([])
        for i in range(num_characteristic_points):
            utility_variables[c].append(LpVariable(f"w_{c}_{i}", lowBound=0, upBound=None))
        utility_thresholds.append(np.linspace(data[:,c].min(), data[:,c].max(), num_characteristic_points))

    # objective function
    prob += epsilon, "Profit"

    # reference ranking

    def utility(c, x):
        tu = 1
        while x > utility_thresholds[c][tu]:
            tu += 1
        return utility_variables[c][tu - 1] + ( (x - utility_thresholds[c][tu - 1]) / (utility_thresholds[c][tu] - utility_thresholds[c][tu - 1]) ) * (utility_variables[c][tu] - utility_variables[c][tu - 1])


    for e in reference_ranking:
        if e[1] == "=":
            prob += lpSum([utility(c, data[e[0], c]) for c in range(data.shape[1])]) == lpSum([utility(c, data[e[2], c]) for c in range(data.shape[1])])
        elif e[1] == ">":
            prob += lpSum([utility(c, data[e[0], c]) for c in range(data.shape[1])]) >= lpSum([utility(c, data[e[2], c]) for c in range(data.shape[1])]) + epsilon
        elif e[1] == "<":
            prob += lpSum([utility(c, data[e[2], c]) for c in range(data.shape[1])]) >= lpSum([utility(c, data[e[0], c]) for c in range(data.shape[1])]) + epsilon

    # normalization
    best_indices = list(map(lambda d : -1 if d == 'gain' else 0, criteria_direction))
    worst_indices = list(map(lambda d : 0 if d == 'gain' else -1, criteria_direction))
    prob += (lpSum([utility_variables[c][best_indices[c]] for c in range(data.shape[1])]) == 1, "upper_normalization")
    prob += (lpSum([utility_variables[c][worst_indices[c]] for c in range(data.shape[1])]) == 0, "lower_normalization")

    # monotonicity
    for c in range(data.shape[1]):
        if criteria_direction[c] == 'gain':
            for i in range(1, len(utility_variables[c])):
                prob += utility_variables[c][i-1] <= utility_variables[c][i]
        else:
            for i in range(1, len(utility_variables[c])):
                prob += utility_variables[c][i-1] >= utility_variables[c][i]

    prob.solve()

    # partial utility functiosn visualization (tmp)
    ws = []
    for c in range(data.shape[1]):
        ws.append([])
        for i in range(num_characteristic_points):
            ws[c].append(utility_variables[c][i].varValue)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    for c in range(data.shape[1]):
        plt.cla()
        plt.plot(utility_thresholds[c], ws[c])
        plt.savefig(f"plots/plot{c}.png")
            

main()


