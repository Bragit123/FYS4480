import numpy as np
import matplotlib.pyplot as plt


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



gs = np.linspace(-1, 1, 101)

def plot_eigvals(eigvals, filename = None, title=None):


    plt.plot(gs, eigvals)
    plt.xlabel("g")
    plt.ylabel("Eigenvalues")

    if title is not None:
        plt.title(title)


    for g_index in [0, 25, 50, 75, 100]:
        # Add the point and annotation box
        g_value = gs[g_index]  # Extract the x-coordinate from gs using g_index
        eigenvalue_at_point = eigvals[g_index, 0]  # Extract the y-coordinate from eigvals_FCI using g_index

        # Plot the point
        plt.plot(g_value, eigenvalue_at_point, 'ro')  # 'ro' for a red point

        # Annotate the point with a box
        plt.annotate(
            f"({g_value:.2f}, {eigenvalue_at_point:.2f})",
            (g_value, eigenvalue_at_point),
            textcoords="offset points",
            xytext=(10, 10),
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")
        )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()



def main():
    n_points = 101
    eigvals_FCI = np.zeros((n_points, 6))
    eigvals_CI = np.zeros((n_points, 5))

    for i, g in enumerate(gs):
        H = hamiltonian(g)
        unsorted_FCI_eigvals = np.linalg.eigvalsh(H)
        eigvals_FCI[i, :] = np.sort(unsorted_FCI_eigvals)

    for i, g in enumerate(gs):
        H_CI = hamiltonian(g)[:5, :5]
        unsorted_CI_eigvals = np.linalg.eigvalsh(H_CI)
        eigvals_CI[i, :] = np.sort(unsorted_CI_eigvals)

    plot_eigvals(eigvals_FCI, "eigvals_FCI.png", "FCI Eigenvalues")
    plot_eigvals(eigvals_CI, "eigvals_CI.png", "CI Eigenvalues")

    print(eigvals_FCI)

    # plt.plot(gs, eigvals_FCI[:, 0], label="FCI")
    # plt.plot(gs, eigvals_CI[:, 0], label="CI")
    plt.plot(gs, np.abs(eigvals_FCI[:, 0] - eigvals_CI[:, 0]) / np.abs(eigvals_FCI[:, 0]), label="|FCI - CI| / |FCI|")
    plt.xlabel("g")
    plt.ylabel("Relative error")
    plt.title("Relative Error in Ground State Energy (FCI vs CI)")
    plt.savefig("eigvals_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()