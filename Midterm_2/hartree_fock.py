
import numpy as np

from matrix_elements import antisym
from states import State1P, get_all_states


class HartreeFock:
    def __init__(self, states: list[State1P], fermi_level: int, g: float):
        self.states = states
        self.fermi_level = fermi_level
        self.g = g

        self.one_particle_integrals = np.diag(
            [state.single_particle_energy() for state in self.states]
            )
        


    def antisymmetrized_matrix(self) -> np.ndarray:
        states = self.states
        A = len(states)
        
        antisymmetrized_matrix = np.zeros((A, A, A, A))
        for a in range(A):
            for b in range(A):
                for c in range(A):
                    for d in range(A):
                        antisymmetrized_matrix[a, b, c, d] = antisym(self.g, states[a], states[b], states[c], states[d])
        return antisymmetrized_matrix

    def hartree_fock_potential(self, C: np.ndarray) -> np.ndarray:
        fermilevel = self.fermi_level 
        C_trunc = C[:fermilevel, :]
        return np.einsum('jc, jd, acbd -> ab', C_trunc, C_trunc, self.antisymmetrized_matrix())

    
    def hartree_fock_matrix(self, C: np.ndarray) -> np.ndarray:
        return self.one_particle_integrals + self.hartree_fock_potential(C)
    

    def iteration(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hf_matrix = self.hartree_fock_matrix(C)
        eigvals, eigvecs = np.linalg.eigh(hf_matrix)
        C = eigvecs.T
        return eigvals, C
    
    def solve(self, n_iters: int, tol: float = 1e-10) -> tuple[list[np.ndarray], list[np.ndarray]]:
        Cs = []
        eigvals_list = []

        C = np.eye(len(self.states))
        Cs.append(C)
        for i in range(n_iters):
            eigvals, C = self.iteration(C)
            Cs.append(C)
            eigvals_list.append(eigvals)

            if i == 0:
                continue
            elif sum(np.abs(eigvals_list[-1] - eigvals_list[-2])) < tol:
                print(f"Converged in {i+1} iterations.")
                break
        else: # If loop completes without breaking.
            print("Did not converge.")
        return eigvals_list, Cs
    
    def energy(self, C: np.ndarray) -> float:

        Z = self.fermi_level
        C_trunc = C[:Z, :]

        term1 = np.einsum('ia, ib, ab -> ', C_trunc, C_trunc, self.one_particle_integrals)

        antisym = self.antisymmetrized_matrix()
        term2 = np.einsum(
            'ia, jb, ic, jd, abcd -> ',
            C_trunc, # from bra i
            C_trunc, #from bra  j
            C_trunc, #from bra i
            C_trunc, # from ket j
            antisym
        )

        return term1 + term2 / 2


def get_hartree_fock_eigenvalues(g_mesh: np.ndarray):
    states = get_all_states()
    fermi_level = 4

    eigenvalues = np.zeros((len(g_mesh), len(states)))
    eigvecs = np.zeros((len(g_mesh), len(states), len(states)))
    
    for i_g, g in enumerate(g_mesh):
        solver = HartreeFock(states, fermi_level, g)
        eigvals_list, Cs = solver.solve(5)
        eigenvalues[i_g] = eigvals_list[-1]
        # eigvecs[i_g] = Cs[-1]

    return eigenvalues, eigvecs

def get_hartree_fock_energy(g_mesh: np.ndarray):
    states = get_all_states()
    fermi_level = 4

    energies = np.zeros(len(g_mesh))
    
    for i_g, g in enumerate(g_mesh):
        solver = HartreeFock(states, fermi_level, g)
        eigvals_list, Cs = solver.solve(5)
        energies[i_g] = solver.energy(Cs[-1])

    return energies


def sandbox():
    states = [
        State1P(1, 1),
        State1P(1, -1),
        State1P(2, 1),
        State1P(2, -1),
        State1P(3, 1),
        State1P(3, -1),
        State1P(4, 1), 
        State1P(4, -1),
    ]

    g = 0
    solver = HartreeFock(states, 4, g)
    eigvals_list, Cs = solver.solve(5)
    # print(solver.energy(Cs[-1]))
    print(eigvals_list[-1])
    





def main():
    sandbox()


if __name__ == "__main__":
    main()


def old():
    pass
    # g = 1
    # solver = HartreeFock(states, 4, g)
    # eigvals_list, Cs = solver.solve(5)


    # # print(solver.energy(Cs[-1]))
    # # print(Cs[-1])

    # # gs = np.linspace(-1, 1, 101)
    # # gs_energies = []
    # # for i, g in enumerate(gs):
    # #     solver = HartreeFock(states, 4, g)
    # #     eigvals_list, Cs = solver.solve(5)
    # #     gs_energies.append(solver.energy(Cs[-1]))


    # # import matplotlib.pyplot as plt
    # # plt.plot(gs, gs_energies)
    # # plt.show()