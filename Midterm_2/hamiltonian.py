import numpy as np


def hamiltonian(g: float) -> np.ndarray:
    E0ref = np.eye(6) * (2 - g)
    F_N = np.array([
        [0, 0, 0, 0, 0, 0, ],
        [0, g + 4, 0, 0, 0, 0],
        [0, 0, g + 6, 0, 0, 0],
        [0, 0, 0, g + 2, 0, 0],
        [0, 0, 0, 0, g + 4, 0],
        [0, 0, 0, 0, 0, 8 + 2*g, ],
    ])
    V_N = np.array([
        [0, -1/2 * g, -1/2 * g, -1/2 * g, -1/2 * g, 0],
        [-1/2 * g, -g, -1/2 * g, -1/2 * g, 0, -1/2 * g, ],
        [-1/2 * g, -1/2 * g, -g, 0, -1/2 * g, -1/2 * g, ],
        [-1/2 * g, -1/2 * g, 0, -g, -1/2 * g, -1/2 * g, ],
        [-1/2 * g, 0, -1/2 * g, -1/2 * g, -g, -1/2 * g, ],
        [0, -1/2 * g, -1/2 * g, -1/2 * g, -1/2 * g, -2*g, ],
    ])
    H = E0ref + F_N + V_N
    return H


