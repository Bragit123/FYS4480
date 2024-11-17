from typing import Literal


class State1P:
    def __init__(self, p: int, sign: Literal[1, -1]):
        self.p = p
        self.sign = sign

    def single_particle_energy(self) -> float:
        return self.p - 1
    

def get_all_states() -> list[State1P]:
    return [
        State1P(1, 1),
        State1P(1, -1),
        State1P(2, 1),
        State1P(2, -1),
        State1P(3, 1),
        State1P(3, -1),
        State1P(4, 1), 
        State1P(4, -1),
    ]



def get_hole_states() -> list[State1P]:
    return [
        State1P(1, -1),
        State1P(1, 1),
        State1P(2, -1),
        State1P(2, 1),
    ]

def get_particle_states() -> list[State1P]:
    return [
        State1P(3, -1),
        State1P(3, 1),
        State1P(4, -1),
        State1P(4, 1),
    ]