import csv
import numpy as np
import os

import os

from UTA_Solver import UTA_Solver
from UTA_GMS_Solver import UTA_GMS_Solver

def main():
    data_file = 'Nuclear waste management.csv'
    data = read_data(data_file)

    criteria_direction = ['cost', 'cost', 'cost', 'cost']

    reference_ranking = [
        [12, '<', 19],
        [7,  '>', 21],
        [4, '<', 7],
        [13, '>', 23],
        [20,  '>', 21]
    ]

    # change to 0-based indexing 
    for ref in reference_ranking:
        ref[0] -= 1
        ref[2] -= 1

    # num_characteristic_points = 3

    # solver =  UTA_Solver(data, criteria_direction, reference_ranking, num_characteristic_points)

    # solver.solve()

    # ranking = solver.rank()

    # ranking_str = ""
    # for i in range(len(ranking)):
    #     ranking_str += f"{(i+1):02d}. {ranking[i][0]+1} ({ranking[i][1]}) \n"

    # solver.plot_partial_utility()

    # os.makedirs("results", exist_ok=True)
    # with open("results/ranking.txt", 'w') as file:
    #     file.write(ranking_str)

    solver = UTA_GMS_Solver(data, criteria_direction, reference_ranking)
    solver.solve()

    solver.hasse_diagram()



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


            

main()


