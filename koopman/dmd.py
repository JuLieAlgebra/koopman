"""Dynamic mode decomposition demo."""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import simulate


def dmd(
    state_history: List[np.array], r: int = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Implements SVD-based Dynamic Mode Decomposition (DMD).

    Given a state history of the system, this function computes the DMD matrix and its corresponding eigenvalues
    and eigenvectors. The DMD provides a low-dimensional approximation of the system dynamics and can be used for
    forecasting, mode analysis, and control.

    Args:
        state_history: A list of numpy arrays representing the state history of the system.
            Each numpy array represents the (flattened) state at a specific time step. The list should be ordered in time,
            where `state_history[0]` represents the state at time t = 0 and `state_history[-1]` represents the
            state at the final time step.
        r: The rank truncation parameter for the DMD matrix. It determines the number of dominant modes to retain.
            If not specified (None), the full-rank DMD matrix is computed.

    Returns:
        A tuple containing the computed DMD matrix, eigenvalues, and eigenvectors.
                - koopman_matrix: The computed Koopman matrix representing the linear dynamics of the system.
                - eigenvalues: The eigenvalues of the Koopman matrix.
                - eigenvectors: The eigenvectors of the Koopman matrix.

    Notes:
        - The state history should have at least two time steps for DMD computation.
        - The first time step in `state_history` corresponds to the initial condition, and the subsequent time steps
          represent the evolution of the system.
    """
    state_history = np.array(state_history)
    last_state = state_history[:-1].copy()
    state = state_history[1:].copy()

    U, S, V = np.linalg.svd(last_state, full_matrices=False)

    koopman_matrix = (
        U[:, :r].conj().T @ state @ V[:r, :].conj() @ np.diag(np.reciprocal(S[:r]))
    )

    eigenvalues, eigenvectors = np.linalg.eig(koopman_matrix)

    return koopman_matrix, eigenvalues, eigenvectors


def visualize_dmd_output(
    state_history: np.ndarray,
    koopman_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> None:
    """
    Visualizes the output of the dynamic mode decomposition (DMD).

    Args:
        state_history: An array of shape (N, D) representing the state history, where N is the number
            of time steps and D is the dimensionality of the state.
        koopman_matrix: The computed Koopman matrix.
        eigenvalues: The eigenvalues of the Koopman matrix.
        eigenvectors: The eigenvectors of the Koopman matrix.

    Returns:
        None (Displays the plots).
    """

    # Extract the real and imaginary parts of the eigenvalues
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    # Plot the eigenvalues in the complex plane
    plt.figure(figsize=(8, 6))
    plt.scatter(real_parts, imag_parts, c="b", marker="o", label="Eigenvalues")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalues in the Complex Plane")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the original and reconstructed state trajectories
    num_time_steps = len(state_history)
    time_steps = range(num_time_steps)  # Adjusted the range to start from 1

    original_states = np.array(state_history)
    reconstructed_states = np.dot(eigenvectors, koopman_matrix)
    print(reconstructed_states)
    print(reconstructed_states.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, original_states[:, 0], "b-", label="Dimension 1 (Original)")
    plt.plot(time_steps, original_states[:, 1], "r-", label="Dimension 2 (Original)")
    plt.xlabel("Time Steps")
    plt.ylabel("State Value")
    plt.title("Original State Trajectories")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        time_steps,
        reconstructed_states[:, 0],
        "b--",
        label="Dimension 1 (Reconstructed)",
    )
    plt.plot(
        time_steps,
        reconstructed_states[:, 1],
        "r--",
        label="Dimension 2 (Reconstructed)",
    )
    plt.xlabel("Time Steps")
    plt.ylabel("State Value")
    plt.title("Reconstructed State Trajectories")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # # Example usage for multivariate time series.
    # timeseries = simulate.TimeSeries()

    # steps = 100
    # timeseries.forward(steps)

    # A, eigenvalues, eigenvectors = dmd(timeseries.state_history)

    # print("Koopman operator:\n", A)
    # print("Eigen values:\n", eigenvalues)
    # print("Eigen vectors:\n", eigenvectors)
    # Example usage
    state_history = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]]
    r = 2

    koopman_matrix, eigenvalues, eigenvectors = dmd(state_history, r)
    visualize_dmd_output(state_history, koopman_matrix, eigenvalues, eigenvectors)
