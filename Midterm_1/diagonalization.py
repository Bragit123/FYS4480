from typing import Literal, Union

import numpy as np

from states import OneParticleOneHole, ReferenceState, ReferenceStateBeryllium, ReferenceStateHelium, State1P, State2P, below_fermi_helium, below_fermi_beryllium


def delta(state1: State1P, state2: State1P):
    return state1 == state2
    

class Diagonalize:
    def __init__(self, system: Literal["helium", "beryllium"] = "helium"):
        if system == "helium":
            self.initialize_helium()
        elif system == "beryllium":
            self.initialize_beryllium()
        else:
            raise ValueError("Invalid system")

    def initialize_helium(self) -> None:
        self.Z = 2

        oneup = State1P(1, 1)
        onedown = State1P(1, -1)
        twoup = State1P(2, 1)
        twodown = State1P(2, -1)
        threeup = State1P(3, 1)
        threedown = State1P(3, -1)

        self.states = [
            ReferenceStateHelium(),
            OneParticleOneHole(twoup, oneup),
            OneParticleOneHole(twodown, onedown),
            OneParticleOneHole(threeup, oneup),
            OneParticleOneHole(threedown, onedown),
        ]

        self.below_fermi = below_fermi_helium()


    def initialize_beryllium(self) -> None:
        self.Z = 4

        oneup = State1P(1, 1)
        onedown = State1P(1, -1)
        twoup = State1P(2, 1)
        twodown = State1P(2, -1)
        threeup = State1P(3, 1)
        threedown = State1P(3, -1)

        self.states = [
            ReferenceStateBeryllium(),
            OneParticleOneHole(threeup, oneup),
            OneParticleOneHole(threeup, twoup),
            OneParticleOneHole(threedown, onedown),
            OneParticleOneHole(threedown, twodown),
        ]

        self.below_fermi = below_fermi_beryllium()

    def bra_fhat_ket(self, state1: State1P, state2: State1P):
        term1 = state1.single_particle_energy(self.Z) * delta(state1, state2)
        
        l1 = [State2P(state1, other_state) for other_state in self.below_fermi]
        l2 = [State2P(state2, other_state) for other_state in self.below_fermi]
        term2 = sum(
            alphabeta.get_matrix_element_AS(gammadelta, self.Z) for alphabeta, gammadelta in zip(l1, l2)
        )   

        return term1 + term2
    
    def matelem_ref_1p1h(self, ref: ReferenceState, state: OneParticleOneHole):
        a = state.particle
        i = state.hole

        return self.bra_fhat_ket(i, a)

    def matelem_1p1h_1p1h(self, state1: OneParticleOneHole, state2: OneParticleOneHole):
        a, i = state1.particle, state1.hole
        b, j = state2.particle, state2.hole

        term1 = self.bra_fhat_ket(a, b) * delta(i, j)
        term2 = - self.bra_fhat_ket(j, i) * delta(a, b)

        ja = State2P(j, a)
        bi = State2P(b, i)
        term3 = ja.get_matrix_element_AS(bi, self.Z)

        refstate = self.states[0]
        term4 =  delta(a, b) * delta(i, j) * refstate.energy()
        return term1 + term2 + term3 + term4

    def matelem(
        self, 
        state1: Union[ReferenceState, OneParticleOneHole],
        state2: Union[ReferenceState, OneParticleOneHole]
    ):
        if isinstance(state1, ReferenceState) and isinstance(state2, ReferenceState):
            return state1.energy()
        elif isinstance(state1, OneParticleOneHole) and isinstance(state2, OneParticleOneHole):
            return self.matelem_1p1h_1p1h(state1, state2)
        elif isinstance(state1, OneParticleOneHole) and isinstance(state2, ReferenceState):
            return self.matelem_ref_1p1h(state2, state1)
        elif isinstance(state1, ReferenceState) and isinstance(state2, OneParticleOneHole):
            return self.matelem_ref_1p1h(state1, state2)
        else:
            raise ValueError("Invalid states")


    def assemble_hamiltonian(self) -> None:
        A = len(self.states)
        H = np.zeros((A, A))
        for i in range(A):
            for j in range(A):
                H[i, j] = self.matelem(self.states[i], self.states[j])
        return H

    def diagonalize(self) -> tuple[np.ndarray, np.ndarray]:
        H = self.assemble_hamiltonian()
        return np.linalg.eigh(H)




if __name__ == "__main__":
    systemhelium = Diagonalize("helium")
    eigvals, eigvecs = systemhelium.diagonalize()
    print("Helium GS energy estimate:", eigvals.min())

    systemberyllium = Diagonalize("beryllium")
    eigvals, eigvecs = systemberyllium.diagonalize()
    print("Beryllium GS energy estimate:", eigvals.min())