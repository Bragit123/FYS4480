
import matplotlib.pyplot as plt
import numpy as np

from diagonalize import get_ground_state_energy_FCI, get_ground_state_energy_CI
from hartree_fock import get_hartree_fock_eigenvalues, get_hartree_fock_energy
from perturbation_theory import perturbation_energy_3rd_order, perturbation_energy_4th_order


def comparison_RSPT_vs_FCI():
    g_mesh = np.linspace(-1, 1, 101)
    ground_state_energy_FCI = get_ground_state_energy_FCI(g_mesh)

    perturbation_energies_3rd = [perturbation_energy_3rd_order(g) for g in g_mesh]
    perturbation_energies_4th = [perturbation_energy_4th_order(g) for g in g_mesh]


    plt.plot(g_mesh, ground_state_energy_FCI, label="FCI")
    plt.plot(g_mesh, perturbation_energies_3rd, label="3rd order RSPT")
    plt.plot(g_mesh, perturbation_energies_4th, label="4th order RSPT")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Ground State Energy Estimate, RSPT vs FCI")
    plt.savefig("perturbation_energy.png")
    plt.close()

    plt.plot(g_mesh, np.abs(ground_state_energy_FCI - perturbation_energies_3rd) / np.abs(ground_state_energy_FCI), label="|FCI - 3rd order RSPT| / |FCI|")
    plt.plot(g_mesh, np.abs(ground_state_energy_FCI - perturbation_energies_4th) / np.abs(ground_state_energy_FCI), label="|FCI - 4th order RSPT| / |FCI|")
    plt.xlabel("g")
    plt.ylabel("Relative error")
    plt.legend()
    plt.title("Relative Error in Ground State Energy (RSPT)")
    plt.savefig("perturbation_energy_error.png")
    plt.close()


def comparison_RSPT_3rd_order_vs_CI_vs_FCI():
    g_mesh = np.linspace(-1, 1, 101)
    ground_state_energy_CI = get_ground_state_energy_CI(g_mesh)
    ground_state_energy_FCI = get_ground_state_energy_FCI(g_mesh)
    ground_state_energy_RSPT_3rd_order = [perturbation_energy_3rd_order(g) for g in g_mesh]

    plt.plot(g_mesh, ground_state_energy_FCI, label="FCI")
    plt.plot(g_mesh, ground_state_energy_CI, label="CI")
    plt.plot(g_mesh, ground_state_energy_RSPT_3rd_order, label="3rd order RSPT")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Ground State Energy, RSPT vs CI vs FCI")
    plt.savefig("perturbation_energy_vs_CI_vs_FCI.png")
    plt.close()

    rel_error_CI = np.abs(ground_state_energy_CI - ground_state_energy_FCI) / np.abs(ground_state_energy_FCI)
    rel_error_RSPT = np.abs(ground_state_energy_RSPT_3rd_order - ground_state_energy_FCI) / np.abs(ground_state_energy_FCI)
    plt.plot(g_mesh, rel_error_CI, label="|CI - FCI| / |FCI|")
    plt.plot(g_mesh, rel_error_RSPT, label="|RSPT - FCI| / |FCI|")
    plt.xlabel("g")
    plt.ylabel("Relative error")
    plt.legend()
    plt.title("Relative Error in Ground State Energy (RSPT vs CI)")
    plt.savefig("perturbation_energy_error_vs_CI.png")
    plt.close()

    g_mesh = np.linspace(0.8, 0.9, 101)
    ground_state_energy_CI = get_ground_state_energy_CI(g_mesh)
    ground_state_energy_FCI = get_ground_state_energy_FCI(g_mesh)
    ground_state_energy_RSPT_3rd_order = [perturbation_energy_3rd_order(g) for g in g_mesh]

    plt.plot(g_mesh, ground_state_energy_FCI, label="FCI")
    plt.plot(g_mesh, ground_state_energy_CI, label="CI")
    plt.plot(g_mesh, ground_state_energy_RSPT_3rd_order, label="3rd order RSPT")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Ground State Energy, RSPT vs CI vs FCI (Zoomed in)")
    plt.savefig("perturbation_energy_vs_CI_vs_FCI_zoomed.png")
    plt.close()


def plot_hartree_fock_eigenvalues():
    g_mesh = np.linspace(-1, 1, 101)
    eigenvalues, eigvecs = get_hartree_fock_eigenvalues(g_mesh)
    plt.plot(g_mesh, eigenvalues)
    plt.xlabel("g")
    plt.ylabel("Eigenvalue")
    plt.title("Hartree-Fock Eigenvalues")
    plt.savefig("hartree_fock_eigenvalues.png")
    plt.close()


def comparison_FCI_vs_HF():
    g_mesh = np.linspace(-1, 1, 101)
    ground_state_energies_FCI = get_ground_state_energy_FCI(g_mesh)
    ground_state_energies_HF = get_hartree_fock_energy(g_mesh)

    plt.plot(g_mesh, ground_state_energies_FCI, label="FCI")
    plt.plot(g_mesh, ground_state_energies_HF, label="HF")
    plt.title("Hartree-Fock Energy vs Exact Energy")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig("FCI_vs_HF.png")
    plt.close()


def compare_CI_coefficients():
    from perturbation_theory import get_RSPT_wavefunction_approximation
    from diagonalize import get_eigvecs_CI, get_eigvecs_FCI

    g_mesh = np.linspace(-1, 1, 101)
    psis_RS = np.array([get_RSPT_wavefunction_approximation(g) for g in g_mesh])
    psis_CI = get_eigvecs_CI(g_mesh)
    psis_FCI = get_eigvecs_FCI(g_mesh)
    

    i = 3
    # psis_CI = np.sign(psis_CI[:, ]) * psis_CI
    plt.plot(g_mesh, psis_RS[:, i], label=f"RSPT {i}")
    plt.plot(g_mesh, psis_CI[:, i, 0], label=f"CI {i}")
    plt.plot(g_mesh, psis_FCI[:, i, 0], label=f"FCI {i}")

    plt.legend()

    plt.show()
    # # standardize sign
    # psis_CI = np.sign(psis_CI[0, 0, :]) * psis_CI

    # for i in range(5):
    #     plt.plot(g_mesh, psis_RS[:, i], label=f"RSPT {i}")
    #     plt.plot(g_mesh, psis_CI[:, i, 0], label=f"CI {i}")

    #     plt.show()

    # g = 1
    # psi_RSPT = get_RSPT_wavefunction_approximation(g)
    # psi_CI = get_eigvecs_CI(np.array([g]))[0, :, 0]

    # print(psi_RSPT)
    # print(psi_CI)

def main():
    # plot_hartree_fock_eigenvalues()
    # comparison_FCI_vs_HF()
    # comparison_RSPT_vs_FCI()
    # comparison_RSPT_3rd_order_vs_CI_vs_FCI()
    compare_CI_coefficients()

if __name__ == "__main__":
    main()

