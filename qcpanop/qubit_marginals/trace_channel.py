"""
Implementation of a qubit trace operation

given rho_{AB}
trace_{B}[rho_{AB}] = \sum_{b}(I \otimes <b|)\rho_{AB}( I\otimes |b>)
"""
import numpy as np


def make_basis_state(ii, dim):
    state = np.zeros((dim, 1), dtype=np.complex128)
    state[ii, 0] = 1
    return state

def main():
    psi_a = np.random.randn(2) + 1j * np.random.randn(2) 
    psi_a /= np.linalg.norm(psi_a)
    psi_a = psi_a.reshape((-1, 1))
    psi_b = np.random.randn(2) + 1j * np.random.randn(2) 
    psi_b /= np.linalg.norm(psi_b)
    psi_b = psi_b.reshape((-1, 1))
    rho_a = psi_a @ psi_a.conj().T
    rho_b = psi_b @ psi_b.conj().T
    rho = np.kron(rho_a, rho_b)
    print(np.trace(rho_a), np.trace(rho_b), np.trace(rho))

    # now trace out B 
    kraus_operators_trace_b = []
    kraus_operators_trace_a = []
    for ii in range(2):
        ii_b = make_basis_state(ii, 2)
        kraus_b = np.kron(np.eye(2), ii_b)
        kraus_a = np.kron(ii_b, np.eye(2))
        kraus_operators_trace_b.append(kraus_b)
        kraus_operators_trace_a.append(kraus_a)

    # sum_{i}K_{i}^ K_{i} = I
    kraus_identity = sum([xx @ xx.conj().T for xx in kraus_operators_trace_b])
    assert np.allclose(kraus_identity, np.eye(4))

    # sum_{i}K_{i}^ K_{i} = I
    kraus_identity = sum([xx @ xx.conj().T for xx in kraus_operators_trace_a])
    assert np.allclose(kraus_identity, np.eye(4))

    rho_a_test = sum([xx.conj().T @ rho @ xx for xx in kraus_operators_trace_b])
    assert np.allclose(rho_a_test, rho_a)

    rho_b_test = sum([xx.conj().T @ rho @ xx for xx in kraus_operators_trace_a])
    assert np.allclose(rho_b_test, rho_b)


    # show that the dual condition yields non-physical states
    rho_b = np.array([[1, 0], [0, -1]])
    rho = np.kron(rho_a, rho_b)
    print(np.trace(rho_a), np.trace(rho_b), np.trace(rho))

    rho_a_test = sum([xx.conj().T @ rho @ xx for xx in kraus_operators_trace_b])
    # recall the partial trace is there
    assert np.allclose(rho_a_test * np.trace(rho_b), rho_a * np.trace(rho_b))

    rho_b_test = sum([xx.conj().T @ rho @ xx for xx in kraus_operators_trace_a])
    assert np.allclose(rho_b_test, rho_b)





if __name__ == "__main__":
    main()