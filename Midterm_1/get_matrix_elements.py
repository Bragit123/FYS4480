import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("matrix_elements.txt", sep=" ", usecols=(0,2))
data.columns = [
    "M_elem",
    "expression"
]
# print(data.head(5))
# Z = 2
# for i in range(5):
#     res = eval(data["expression"][i])
#     print(res)


def get_matrix_element_expression(alphabeta: tuple[int, int], gammadelta: tuple[int, int]) -> str:
    alpha, beta = alphabeta
    gamma, delta = gammadelta
    string = f"<{alpha}{beta}|V|{gamma}{delta}>"
    row = data[data["M_elem"] == string]
    expression = row["expression"].values[0]
    return expression

def matrix_element_nospin(Z: int, alphabeta: tuple[int, int], gammadelta: tuple[int, int]) -> float:
    expression = get_matrix_element_expression(alphabeta, gammadelta)
    return eval(expression)


# print(matrix_element_nospin(1, 1, 1, 1, 1))


# print(get_matrix_element_expression(1, 2, 1, 1))
# # print(data["M_elem"][0], len(data["M_elem"][0]))


# string = "<11|V|11>"
# row = data[data["M_elem"] == string]
# print(row)