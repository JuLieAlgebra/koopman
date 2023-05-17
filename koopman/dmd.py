"""Dynamic mode decomposition demo."""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import simulate


def dmd(
    state_history: List[np.array], r: int = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Implements SVD-based dynamic mode decomposition.

    Assumes state_history[0] is an np.array of the state at time t = 0 and state_history[-1] grabs the the last
    state.
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
