import numpy as np

from states import State1P, State2P, below_fermi_helium, below_fermi_beryllium, get_all_1p_states


class HartreeFock:
    def __init__(self, below_fermi_states: list[State1P], Z: int):
        self.states = below_fermi_states
        self.Z = Z

        self.one_particle_integrals = np.diag(
            [state.single_particle_energy(Z) for state in self.states]
            )

    def antisymmetrized_matrix(self) -> np.ndarray:
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
        fermilevel = self.Z 
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

        Z = self.Z
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
        # # Compute the Hartree-Fock energy
        # E_HF = 0.5 * np.sum(D * (F))

        # return E_HF
            


def problem_f():
    Zs = [2, 4]
    atom_names = ["Helium", "Beryllium"]
    for Z, name in zip(Zs, atom_names):
        states = get_all_1p_states()
        solver = HartreeFock(states, Z)
        C0 = np.eye(len(states))
        eigvals, C1 = solver.iteration(C0)
        ground_state_energy = solver.energy(C1)
        print(f"Single-particle energies, {name}, after one iteration:")
        print(eigvals)

        print(f"Ground state energy, {name}, after one iteration:")
        print(ground_state_energy)
        

def problem_g():
    Zs = [2, 4]
    atom_names = ["Helium", "Beryllium"]
    for Z, name in zip(Zs, atom_names):
        states = get_all_1p_states()
        solver = HartreeFock(states, Z)
        n_iters = 1000
        eigvals_list, C_list = solver.solve(n_iters)
        ground_state_energy = solver.energy(C_list[-1])
        print(f"Single-particle energies, {name}:")
        print(eigvals_list[-1])

        print(f"Ground state energy, {name}:")
        print(ground_state_energy)
    


if __name__ == "__main__":
    print("Problem f:\n")
    problem_f()
    print("\nProblem g:\n")
    problem_g()
