import numpy as np

from states import State1P, State2P, below_fermi_helium, below_fermi_beryllium, get_all_1p_states


class HartreeFock:
    def __init__(self, below_fermi_states: list[State1P], Z: int):
        self.states = below_fermi_states
        self.Z = Z

        self.one_particle_integrals = np.diag(
            [state.single_particle_energy(Z) for state in self.states]
            )

    def antiymmetrized_matrix(self) -> np.ndarray:
        states = self.states
        A = len(states)
        
        antisymmetrized_matrix = np.zeros((A, A, A, A))
        for a in range(A):
            for b in range(A):
                for c in range(A):
                    for d in range(A):
                        alphabeta = State2P(states[a], states[b])
                        gammadelta = State2P(states[c], states[d])
                        antisymmetrized_matrix[a, b, c, d] = alphabeta.get_matrix_element_AS(
                            gammadelta,
                            self.Z,
                        )
        return antisymmetrized_matrix

    def hartree_fock_potential(self, C: np.ndarray) -> np.ndarray:
        antisym = self.antiymmetrized_matrix()
        return np.einsum("jc, jd, acbd -> ab", C, C, antisym)
    
    def hartree_fock_matrix(self, C: np.ndarray) -> np.ndarray:
        return self.one_particle_integrals + self.hartree_fock_potential(C)
    

    def iteration(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = self.hartree_fock_matrix(C)
        eigvals, C = np.linalg.eigh(H)
        return eigvals, C
    
    def solve(self, n_iters):
        Cs = []
        eigvals_list = []

        C = np.eye(len(self.states))
        for i in range(n_iters):
            eigvals, C = self.iteration(C)
            Cs.append(C)
            eigvals_list.append(eigvals)

        return eigvals_list, Cs
    

if __name__ == "__main__":
    Z = 4
    states = get_all_1p_states()
    # states = below_fermi_beryllium()
    solver = HartreeFock(states, Z)


    print(solver.antiymmetrized_matrix())

    # print(solver.one_particle_integrals)
    eigvals_list, Cs = solver.solve(1)

    # eigvals, C = eigvals_list[-1], Cs[-1]
    print(*eigvals_list, sep='\n')
