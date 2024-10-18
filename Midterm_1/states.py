from __future__ import annotations
from typing import Literal

from get_matrix_elements import matrix_element_nospin, matrix_element_nospin_AS


class State1P:
    def __init__(self, n: int, spin: Literal[1, -1]):
        self.n = n
        self.spin = spin

    def __eq__(self, other):
        return self.n == other.n and self.spin == other.spin
    
    def __str__(self):
        return f"(n={self.n}, spin={self.spin})"
    
    def single_particle_energy(self, Z):
        return - Z**2 / (2 * self.n**2)

    
        

class State2P:
    def __init__(self, P1: State1P, P2: State1P):
        self.P1 = P1
        self.P2 = P2

    def __eq__(self, other):
        return self.P1 == other.P1 and self.P2 == other.P2
    
    def same_spins(self):
        return self.P1.spin == self.P2.spin
    
    def get_alphabeta(self):
        return (self.P1.n, self.P2.n)
    
    def get_matrix_element_nospin(self, other: State2P, Z):
        alphabeta = self.get_alphabeta()
        gammadelta = other.get_alphabeta()
        return matrix_element_nospin(Z, alphabeta, gammadelta)
    
    def get_matrix_element_nospin_AS(self, other: State2P, Z):
        alphabeta = self.get_alphabeta()
        gammadelta = other.get_alphabeta()
        return matrix_element_nospin_AS(Z, alphabeta, gammadelta)
    
    def get_matrix_element_AS(self, other: State2P, Z):
        if self.same_spins():
            return self.get_matrix_element_nospin_AS(other, Z)
        else:
            return self.get_matrix_element_nospin(other, Z)
        
    def __str__(self):
        return f"[[{self.P1}, {self.P2})]]"
    

def below_fermi_helium() -> list[State1P]:
    return [State1P(1, 1), State1P(1, -1)]

def below_fermi_beryllium() -> list[State1P]:
    return [State1P(1, 1), State1P(1, -1), State1P(2, 1), State1P(2, -1)]

def get_all_1p_states():
    return [State1P(1, 1), State1P(1, -1), State1P(2, 1), State1P(2, -1), State1P(3, 1), State1P(3, -1)]



if __name__ == "__main__":
    l1 = below_fermi_helium()
    ket1, ket2 = l1
    state = State2P(ket1, ket2)
    print(state.get_alphabeta())
    print(state.get_matrix_element_nospin(state, 2))
    print(state.get_matrix_element_nospin_AS(state, 2))
    print(state.get_matrix_element_AS(state, 2))
    print(state.same_spins())
    print(state)


class OneParticleOneHole():
    def __init__(self, particle: State1P, hole: State1P):
        self.particle = particle
        self.hole = hole

    def __eq__(self, other):
        return self.particle == other.particle and self.hole == other.hole

    def __str__(self):
        return f"({self.particle}, {self.hole})"


class ReferenceState:
    def __init__(self):
        pass

    def energy(self):
        raise NotImplementedError


class ReferenceStateHelium(ReferenceState):
    def __init__(self):
        self.Z = 2
    def energy(self):
        return - self.Z**2  + 5 / 8 * self.Z


class ReferenceStateBeryllium(ReferenceState):
    def __init__(self):
        self.Z = 4
    def energy(self):
        return - 5/4 * self.Z**2  + 586373/373248 * self.Z

# below_fermi_beryllium()









# class Translator:
#     def __init__(self):
#         pass

#     def index_to_state(self, index: int):
#         pass

#     # def state_to_index(self, state) -> int:
#     #     pass

#     def get_spins_equal(self, index: int) -> bool:
#         pass

#     def 