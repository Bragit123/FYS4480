import numpy as np


from matrix_elements import antisym

from states import State1P, get_hole_states, get_particle_states

def epsilon_p(p: int) -> float:
    """Single particle energy of state (p, sigma)."""
    return p - 1


def epsilon_denom(n_hole_states: int, n_particle_states: int) -> np.ndarray:
    """Gives an array with entries of the form
        arr[i, j ,..., a, b] = epsilon_i + epsilon_j + ... - epsilon_a - epsilon_b.
    Hole states have a positive sign, particle states have a negative sign.

    Args:
        n_hole_states (int): Number of hole states.
        n_particle_states (int): Number of particle states.

    Returns:
        np.ndarray: Array with epsilon denominators
    """
    hole_states = get_hole_states()
    particle_states = get_particle_states()

    epsilon_hole = np.array([epsilon_p(state.p) for state in hole_states])
    minus_epsilon_particle = np.array([- epsilon_p(state.p) for state in particle_states])

    shapes = [[1 for _ in range(n_hole_states + n_particle_states)] for _ in range(n_hole_states + n_particle_states)]
    for i in range(n_hole_states):
        shapes[i][i] = len(hole_states)
    for i in range(n_particle_states):
        shapes[i+n_hole_states][i+n_hole_states] = len(particle_states)

    all_eps = []
    for i in range(n_hole_states):
        eps = np.reshape(epsilon_hole, shapes[i])
        all_eps.append(eps)

    for i in range(n_particle_states):
        eps = np.reshape(minus_epsilon_particle, shapes[i+n_hole_states])
        all_eps.append(eps)

    return sum(all_eps)

def get_matelems_AS(
        g: float,
        states_p: list[State1P],
        states_q: list[State1P],
        states_r: list[State1P],
        states_s: list[State1P],
    ) -> np.ndarray:
    """A matrix with all the antisymmetrized matrix elements.

    Args:
        g (float): interaction strength
        states_p (list[State1P]): state.
        states_q (list[State1P]): state.
        states_r (list[State1P]): state.
        states_s (list[State1P]): state.

    Returns:
        np.ndarray: matrix with antisymmetrized matrix elements.
    """
    matelems_AS = np.zeros((len(states_p), len(states_q), len(states_r), len(states_s)))

    for p, state_p in enumerate(states_p):
        for q, state_q in enumerate(states_q):
            for r, state_r in enumerate(states_r):
                for s, state_s in enumerate(states_s):
                    matelems_AS[p, q, r, s] = antisym(g, state_p, state_q, state_r, state_s)
    return matelems_AS


def diagram_1_third_order(g: float):
    """Energy contribution from diagram 1 (3rd order perturbation theory)."""
    a = b = get_particle_states()
    i = j = get_hole_states()

    abij = get_matelems_AS(g, a, b, i, j)
    ijab = get_matelems_AS(g, i, j, a, b)

    eps_ijab = epsilon_denom(2, 2)
    return 1/4 * np.einsum(
        'abij, ijab, ijab -> ', abij, ijab, 1/eps_ijab
    )

def diagram_4_third_order(g: float):
    """Energy contribution from diagram 4 (3rd order perturbation theory)."""
    a = b = c = d = get_particle_states()
    i = j = get_hole_states()

    ijac = get_matelems_AS(g, i, j, a, c)
    acbd = get_matelems_AS(g, a, c, b, d)
    bdij = get_matelems_AS(g, b, d, i, j)
    eps_ijbd = epsilon_denom(2, 2)
    eps_ijac = epsilon_denom(2, 2)

    return 1/8 * np.einsum(
        'ijac, acbd, bdij, ijbd, ijac -> ', ijac, acbd, bdij, 1/eps_ijbd, 1/eps_ijac
    )


def diagram_5_third_order(g: float):
    """Energy contribution from diagram 5 (3rd order perturbation theory)."""
    a = b = get_particle_states()
    i = j = k = l = get_hole_states()

    ikab = get_matelems_AS(g, i, k, a, b)
    jlik = get_matelems_AS(g, j, l, i, k)
    abjl = get_matelems_AS(g, a, b, j, l)
    eps_jlab = epsilon_denom(2, 2)
    eps_ikab = epsilon_denom(2, 2)

    return 1/8 * np.einsum(
        'ikab, jlik, abjl, jlab, ikab -> ', ikab, jlik, abjl, 1/eps_jlab, 1/eps_ikab
    )


def diagram_8_third_order(g: float) -> float:
    """Energy contribution from diagram 8 (3rd order perturbation theory)."""
    a = b = get_particle_states()
    i = j = k = l = get_hole_states()

    ikab = get_matelems_AS(g, i, k, a, b)
    ljli = get_matelems_AS(g, l, j, l, i)
    abjk = get_matelems_AS(g, a, b, j, k)
    eps_ikab = epsilon_denom(2, 2)
    eps_jkab = epsilon_denom(2, 2)

    return -1/2 * np.einsum(
        'ikab, ljli, abjk, ikab, jkab -> ', ikab, ljli, abjk, 1/eps_ikab, 1/eps_jkab
    )


def diagram_5_fourth_order(g: float):
    """Energy contribution from diagram 5 (4th order perturbation theory)."""
    a = b = c = d = get_particle_states()
    i = j = k = l = get_hole_states()

    ikac = get_matelems_AS(g, i, k, a, c)
    jlik = get_matelems_AS(g, j, l, i, k)
    bdjl = get_matelems_AS(g, b, d, j, l)
    acbd = get_matelems_AS(g, a, c, b, d)

    eps_ikac = epsilon_denom(2, 2)
    eps_ikbd = epsilon_denom(2, 2)
    eps_jlbd = epsilon_denom(2, 2)

    return 1 / 16 * np.einsum(
        'ikac, jlik, bdjl, acbd, ikac, ikbd, jlbd -> ', ikac, jlik, bdjl, acbd, 1/eps_ikac, 1/eps_ikbd, 1/eps_jlbd
    )


def diagram_6_fourth_order(g: float):
    """Energy contribution from diagram 6 (4th order perturbation theory)."""
    # diagram 6 is conjugate to diagram 5. 
    # Diagram 5 contribution is real, diagram 6 contribution is same as for 5. 
    return diagram_5_fourth_order(g)


def diagram_14_fourth_order(g: float):
    """Energy contribution from diagram 14 (4th order perturbation theory)."""
    a = b = c = d = e = f = get_particle_states()
    i = j = get_hole_states()

    ijad = get_matelems_AS(g, i, j, a, d)
    adbe = get_matelems_AS(g, a, d, b, e)
    becf = get_matelems_AS(g, b, e, c, f)
    cfij = get_matelems_AS(g, c, f, i, j)

    eps_ijad = epsilon_denom(2, 2)
    eps_ijbe = epsilon_denom(2, 2)
    eps_ijcf = epsilon_denom(2, 2)

    return 1 / 16 * np.einsum(
        'ijad, adbe, becf, cfij, ijad, ijbe, ijcf -> ', ijad, adbe, becf, cfij, 1/eps_ijad, 1/eps_ijbe, 1/eps_ijcf
    )

def diagram_15_fourth_order(g: float):
    """Energy contribution from diagram 15 (4th order perturbation theory)."""
    a = b = get_particle_states()
    i = j = k = l = m = n = get_hole_states()

    ilab = get_matelems_AS(g, i, l, a, b)
    jmil = get_matelems_AS(g, j, m, i, l)
    knjm = get_matelems_AS(g, k, n, j, m)
    abkn = get_matelems_AS(g, a, b, k, n)

    eps_ilab = epsilon_denom(2, 2)
    eps_jmab = epsilon_denom(2, 2)
    eps_knab = epsilon_denom(2, 2)

    return 1/16 * np.einsum(
        'ilab, jmil, knjm, abkn, ilab, jmab, knab -> ', ilab, jmil, knjm, abkn, 1/eps_ilab, 1/eps_jmab, 1/eps_knab
    )

def diagram_36_fourth_order(g: float):
    """Energy contribution from diagram 36 (4th order perturbation theory)."""
    a = b = c = d = get_particle_states()
    i = j = k = l = get_hole_states()
    
    ilac = get_matelems_AS(g, i, l, a, c)
    jkbd = get_matelems_AS(g, j, k, b, d)
    acjk = get_matelems_AS(g, a, c, j, k)
    bdil = get_matelems_AS(g, b, d, i, l)

    eps_ilac = epsilon_denom(2, 2)
    eps_ijklabcd = epsilon_denom(4, 4)
    eps_ilbd = epsilon_denom(2, 2)

    return 1/16 * np.einsum(
        'ilac, jkbd, acjk, bdil, ilac, ijklabcd, ilbd -> ', ilac, jkbd, acjk, bdil, 1/eps_ilac, 1/eps_ijklabcd, 1/eps_ilbd
    )

def diagram_37_fourth_order(g: float):
    """Energy contribution from diagram 37 (4th order perturbation theory)."""
    a = b = c = d = get_particle_states()
    i = j = k = l = get_hole_states()

    ikad = get_matelems_AS(g, i, k, a, d)
    jlbc = get_matelems_AS(g, j, l, b, c)
    bcik = get_matelems_AS(g, b, c, i, k)
    adjl = get_matelems_AS(g, a, d, j, l)

    eps_ikad = epsilon_denom(2, 2)
    eps_ijklabcd = epsilon_denom(4, 4)
    eps_jlad = epsilon_denom(2, 2)

    return 1/16 * np.einsum(
        'ikad, jlbc, bcik, adjl, ikad, ijklabcd, jlad -> ', ikad, jlbc, bcik, adjl, 1/eps_ikad, 1/eps_ijklabcd, 1/eps_jlad
    )



def E0ref(g: float) -> float:
    """Reference energy."""
    return 2 - g 

def perturbation_energy_3rd_order(g: float) -> float:
    """Energy estimate from 3rd order Rayleigh-Schroedinger perturbation theory."""
    return (
        E0ref(g) 
        + diagram_1_third_order(g) 
        + diagram_4_third_order(g) 
        + diagram_5_third_order(g) 
        + diagram_8_third_order(g)
    )

def perturbation_energy_4th_order(g: float):
    """Energy estimate from 4th order Rayleigh-Schroedinger perturbation theory."""
    return (
        E0ref(g)
        + diagram_1_third_order(g)
        + diagram_4_third_order(g)
        + diagram_5_third_order(g)
        + diagram_8_third_order(g)
        + diagram_5_fourth_order(g)
        + diagram_6_fourth_order(g)
        + diagram_14_fourth_order(g)
        + diagram_15_fourth_order(g)
        + diagram_36_fourth_order(g)
        + diagram_37_fourth_order(g)
    )

def get_RSPT_wavefunction_1p1h_coeff(g: float, particle: int, hole: int) -> float:
    """Get one-particle-one-hole coefficient for RSPT wavefunction."""
    if g == 0:
        return 0
    return -1/(2 *g)  * 1/(2 * epsilon_p(hole) - 2 * epsilon_p(particle))
    

def get_RSPT_wavefunction_approximation(g: float):
    """Get approximate RSPT wavefunction."""
    # zero'th coefficient is for reference state.
    # first: particle in 3, hole in 1
    # second: particle in 4, hole in 1
    # third: particle in 3, hole in 2
    # fourth: particle in 4, hole in 2
    psi = np.zeros(5)
    psi[0] = 1
    psi[1] = get_RSPT_wavefunction_1p1h_coeff(g, 3, 1)
    psi[2] = get_RSPT_wavefunction_1p1h_coeff(g, 4, 1)
    psi[3] = get_RSPT_wavefunction_1p1h_coeff(g, 3, 2)
    psi[4] = get_RSPT_wavefunction_1p1h_coeff(g, 4, 2)
    return psi / np.linalg.norm(psi)






if __name__ == "__main__":
    pass