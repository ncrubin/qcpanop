import numpy as np


def get_random_state(n_qubits) -> np.ndarray:
    state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    state /= np.linalg.norm(state)
    return state
