import csv
import numpy as np
import os

import os

from UTA_Solver import UTA_Solver
from UTA_GMS_Solver import UTA_GMS_Solver
from UTA_REP_Solver import UTA_REP_Solver

def main():
    os.makedirs("results", exist_ok=True)

    data_file = 'Nuclear waste management.csv'
    data = read_data(data_file)

    criteria_direction = ['cost', 'cost', 'cost', 'cost']

    reference_ranking = [
        [12, '<', 19],
        [7,  '>', 27],
        [2, '<', 19],
        [5,  '>', 25],
        [1, "<", 11]
    ]

    # change to 0-based indexing 
    for ref in reference_ranking:
        ref[0] -= 1
        ref[2] -= 1

    # UTA

    num_characteristic_points = 4

    solver_UTA =  UTA_Solver(data, criteria_direction, reference_ranking, num_characteristic_points, utility_min_weight=0.1, utility_max_weight=0.5)

    epsilon_UTA = solver_UTA.solve()

    ranking = solver_UTA.rank()

    save_ranking(ranking, "ranking_UTA.txt")

    solver_UTA.plot_partial_utility()

    # UTA GMS

    solver_UTA_GMS = UTA_GMS_Solver(data, criteria_direction, reference_ranking)
    solver_UTA_GMS.solve()

    solver_UTA_GMS.hasse_diagram()

    # Representative utility function

    solver_UTA_REP = UTA_REP_Solver(data, criteria_direction, reference_ranking, solver_UTA_GMS.necessary_relation)
    epsilon_UTA_REP = solver_UTA_REP.solve()
    solver_UTA_REP.plot_partial_utility()

    ranking = solver_UTA_REP.rank()

    save_ranking(ranking, "ranking_UTA_REP.txt")

    with open(os.path.join("results", "epsilon.txt"), 'w') as file:
        file.write(f"UTA: {epsilon_UTA}\n")
        file.write(f"UTA REP: {epsilon_UTA_REP}\n")


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

def save_ranking(ranking, filename, directory="results"):
    ranking_str = ""
    pos = 1
    for i in range(len(ranking)):
        ranking_str += f"{(pos)}.\t{ranking[i][0]+1}\t{ranking[i][1]} \n"
        if i < len(ranking) - 1 and ranking[i][1] != ranking[i+1][1]:
            pos += 1

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as file:
        file.write(ranking_str)
            

main()


