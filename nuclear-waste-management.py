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

    problem, characteristic_thresholds, utility_variables =  construct_LP_problem(data, criteria_direction, reference_ranking, num_characteristic_points)

    target_val, characteristic_values = solve(problem, utility_variables)

    ranking = rank(data, characteristic_thresholds, characteristic_values)

    ranking_str = ""
    for i in range(len(ranking)):
        ranking_str += f"{i:02d}. {ranking[i][0]} ({ranking[i][1]}) \n"

    with open("results/ranking.txt", 'w') as file:
        file.write(ranking_str)

    plot_partial_utility(characteristic_thresholds, characteristic_values)


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

    prob = LpProblem("Maximize_the_epsilon", LpMaximize)

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
        succ_theshold = 1
        while x > utility_thresholds[c][succ_theshold]:
            succ_theshold += 1
        return utility_variables[c][succ_theshold - 1] + ( (x - utility_thresholds[c][succ_theshold - 1]) / (utility_thresholds[c][succ_theshold] - utility_thresholds[c][succ_theshold - 1]) ) * (utility_variables[c][succ_theshold] - utility_variables[c][succ_theshold - 1])


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

    return prob, utility_thresholds, utility_variables

def solve(problem, utility_variables):

    problem.solve()

    utility_variables_values = []
    for c in range(len(utility_variables)):
        utility_variables_values.append([])
        for var in utility_variables[c]:
            utility_variables_values[c].append(var.varValue)

    optimised = value(problem.objective)

    return optimised, utility_variables_values

def rank(data, characteristic_thresholds, characteristic_values):
    ranking = []
    for i in range(data.shape[0]):
        ranking.append((i, utility(data[i], characteristic_thresholds, characteristic_values)))

    ranking.sort(key = lambda x : -x[1])

    return ranking

def partial_utility(x, characteristic_thresholds, characteristic_values):
    succ_theshold = 1
    while x > characteristic_thresholds[succ_theshold]:
        succ_theshold += 1
    return characteristic_values[succ_theshold - 1] + ( (x - characteristic_thresholds[succ_theshold - 1]) / (characteristic_thresholds[succ_theshold] - characteristic_thresholds[succ_theshold - 1]) ) * (characteristic_values[succ_theshold] - characteristic_values[succ_theshold - 1])

def utility(variant, characteristic_thresholds, characteristic_values):
    u = 0
    for i in range(len(variant)):
        u += partial_utility(variant[i], characteristic_thresholds[i],  characteristic_values[i])
    return u

def plot_partial_utility(characteristic_thresholds, characteristic_values, directory="results/plots"):

    os.makedirs(directory, exist_ok=True)

    for c in range(len(characteristic_thresholds)):
        plt.cla()
        plt.plot(characteristic_thresholds[c], characteristic_values[c])
        plt.savefig(os.path.join(directory, f"U_{c}.png"))
            

main()


