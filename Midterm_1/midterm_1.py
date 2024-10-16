import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("matrix_elements.txt", sep=" ", usecols=(0,2))
data.columns = [
    "M_elem",
    "expression"
]
print(data.head(5))
Z = 2
for i in range(5):
    res = eval(data["expression"][i])
    print(res)