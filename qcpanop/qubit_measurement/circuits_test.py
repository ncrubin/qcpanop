from .state_utils import get_random_state
from .circuits import bell_measurement

import numpy as np
import cirq


def test_bell_measurement():
    """Test Bell measurements can simultaneously measure XX, YY, ZZ"""
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    bell_u = cirq.unitary(
        bell_measurement(qubits[0], qubits[1], with_measuremet=False))
    state2q = get_random_state(2)
    state2qc = state2q.reshape((-1, 1))
    state2qcb = bell_u @ state2qc

    XX = np.kron(cirq.PAULI_BASIS['X'], cirq.PAULI_BASIS['X'])
    YY = np.kron(cirq.PAULI_BASIS['Y'], cirq.PAULI_BASIS['Y'])
    ZZ = np.kron(cirq.PAULI_BASIS['Z'], cirq.PAULI_BASIS['Z'])
    ZI = np.kron(cirq.PAULI_BASIS['Z'], np.eye(2))
    IZ = np.kron(np.eye(2), cirq.PAULI_BASIS['Z'])

    xx = cirq.PauliString({qubits[0]: 'X', qubits[1]: 'X'})
    yy = cirq.PauliString({qubits[0]: 'Y', qubits[1]: 'Y'})
    zz = cirq.PauliString({qubits[0]: 'Z', qubits[1]: 'Z'})
    zi = cirq.PauliString({qubits[0]: 'Z'})
    iz = cirq.PauliString({qubits[1]: 'Z'})
    tval1 = xx.expectation_from_state_vector(state2q, qubit_map=qubit_map).real
    tval2 = (state2qc.conj().T @ XX @ state2qc).real
    tval3 = (state2qcb.conj().T @ ZI @ state2qcb).real
    tval4 = zi.expectation_from_state_vector(state2qcb.flatten(), qubit_map=qubit_map).real
    assert np.isclose(tval2, tval1)
    assert np.isclose(tval3, tval1)
    assert np.isclose(tval4, tval1)

    tval1 = yy.expectation_from_state_vector(state2q, qubit_map=qubit_map).real
    tval2 = (state2qc.conj().T @ YY @ state2qc).real
    tval3 = -(state2qcb.conj().T @ ZZ @ state2qcb).real
    tval4 = -zz.expectation_from_state_vector(state2qcb.flatten(), qubit_map=qubit_map).real
    assert np.isclose(tval2, tval1)
    assert np.isclose(tval3, tval1)
    assert np.isclose(tval4, tval1)

    tval1 = zz.expectation_from_state_vector(state2q, qubit_map=qubit_map).real
    tval2 = (state2qc.conj().T @ ZZ @ state2qc).real
    tval3 = (state2qcb.conj().T @ IZ @ state2qcb).real
    tval4 = iz.expectation_from_state_vector(state2qcb.flatten(), qubit_map=qubit_map).real
    assert np.isclose(tval2, tval1)
    assert np.isclose(tval3, tval1)
    assert np.isclose(tval4, tval1)


