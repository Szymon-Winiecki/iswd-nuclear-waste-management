import csv
import numpy as np
import os

import os

from UTA_Solver import UTA_Solver

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

    solver =  UTA_Solver(data, criteria_direction, reference_ranking, num_characteristic_points)

    solver.solve()

    ranking = solver.rank()

    ranking_str = ""
    for i in range(len(ranking)):
        ranking_str += f"{i:02d}. {ranking[i][0]} ({ranking[i][1]}) \n"

    solver.plot_partial_utility()

    os.makedirs("results", exist_ok=True)
    with open("results/ranking.txt", 'w') as file:
        file.write(ranking_str)


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


