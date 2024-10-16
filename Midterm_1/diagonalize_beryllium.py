from typing import Literal

import numpy as np

from get_matrix_elements import matrix_element_nospin




# Define mapping from state to index in Hamiltonian. 
# Spin up is encoded as spin=1, spin down as spin=-1.
def state_to_index(hole_n: Literal[1, 2], spin: Literal[1, -1]) -> int:
    return hole_n + spin + 1

def index_to_state(index: int) -> tuple[Literal[1, 2], Literal[1, -1]]:
    hole_n = (index - 1) % 2 + 1
    spin = (index - 1) // 2 * 2 - 1
    return (hole_n, spin)


def matelem_ref_to_ref(Z: int) -> float:
    return - 5/4 * Z**2  + 586373/373248 * Z


def matelem_ref_to_1h1p(Z: int, hole_n: Literal[1, 2], spin: Literal[1, -1]) -> np.ndarray:
    onebody_part = 0
    twobody_part = None
    raise NotImplementedError


def delta(alpha, beta):
    return int(alpha == beta)


def h0(Z, alpha: tuple[int, int], beta: tuple[int, int]) -> float:
    orbital1, spin1 = alpha
    return - Z**2 / (2 * orbital1**2) * delta(alpha, beta) 


def matelem_1h1p_to_1h1p_onebody_contribution(
    Z: int,
    a: tuple[int, int],
    i: tuple[int, int],
    b: tuple[int, int],
    j: tuple[int, int]
) -> np.ndarray:
    below_fermi = [(1, 1), (1, -1), (2, 1), (2, -1)]

    term1_onebody = h0(Z, a, b) * delta(i, j)
    term2_onebody = h0(Z, i, j) * delta(a, b)
    term3_onebody = h0(Z, b, b) * delta(i, j) 
    term4_onebody = delta(a, b) * delta(i, j) *  sum(h0(Z, k, k) * (1 - delta(i, k))for k in below_fermi)

    return term1_onebody + term2_onebody + term3_onebody + term4_onebody


def matelem_1h1p_to_1h1p_twobody_contribution() -> np.ndarray:
    raise NotImplementedError
        

def matelem_1h1p_to_1h1p(
    Z: int,
    hole_n1: Literal[1, 2],
    spin1: Literal[1, -1],
    hole_n2: Literal[1, 2],
    spin2: Literal[1, -1]
) -> np.ndarray:
    a = (3, spin1)
    i = (hole_n1, spin1)
    b = (3, spin2)
    j = (hole_n2, spin2)

    onebody_contribution = matelem_1h1p_to_1h1p_onebody_contribution(Z, a, i, b, j)
    twobody_contribution = matelem_1h1p_to_1h1p_twobody_contribution()

    return onebody_contribution + twobody_contribution


    

def assemble_hamiltonian_beryllium() -> np.ndarray:
    Z = 4
    H = np.zeros((5, 5))
    H[0, 0] = matelem_ref_to_ref(4)
    for i in range(1, 5):
        H[0, i]  = matelem_ref_to_1h1p(Z, *index_to_state(i))
        H[i, 0]  = matelem_ref_to_1h1p(Z, *index_to_state(i))

    for i in range(1, 5):
        for j in range(1, 5):
            H[i, j] = matelem_1h1p_to_1h1p(Z, *index_to_state(i), *index_to_state(j))



def diagonalize_beryllium() -> tuple[np.ndarray, np.ndarray]:
    H = assemble_hamiltonian_beryllium()
    return np.linalg.eigh(H)


if __name__ == "__main__":
    eigenvalues, eigenvectors = diagonalize_beryllium()
    print("Minimal eigenvalue:", eigenvalues.min())
    print("All eigenvalues:", eigenvalues)
    