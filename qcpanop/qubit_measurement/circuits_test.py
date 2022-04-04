from qcpanop.qubit_measurement.state_utils import get_random_state
from qcpanop.qubit_measurement.circuits import bell_measurement, create_measurement_circuit, zeta_circuit

import numpy as np
import cirq


def test_bell_measurement_mathtest():
    """Test Bell measurements can simultaneously measure XX, YY, ZZ"""
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    bell_u = cirq.unitary(
        bell_measurement(qubits[0], qubits[1], with_measurements=False))
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

def tes_bell_measurement_unitary():
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    bell_u = cirq.unitary(
        zeta_circuit(qubits[1]) + bell_measurement(ancilla_qubit=qubits[1],
                                                   system_qubit=qubits[0],
                                                   with_measurements=False))

    bell_u_test = create_measurement_circuit([qubits[0]],
                                             {qubits[0]: qubits[1]},
                                             compile=False)
    bell_u_test = cirq.unitary(bell_u_test)
    assert np.isclose(abs(np.trace(bell_u_test.conj().T @ bell_u)), 4)

    n_qubits = 4
    qubits = cirq.LineQubit.range(2*n_qubits)
    system_qubits = qubits[:n_qubits]
    ancilla_qubits = qubits[n_qubits:]
    qubit_map = dict(zip(system_qubits, ancilla_qubits))
    measurement_bell = cirq.Circuit()
    for sys_q, anc_q in zip(system_qubits, ancilla_qubits):
        measurement_bell += zeta_circuit(anc_q)
        measurement_bell += bell_measurement(ancilla_qubit=anc_q,
                                             system_qubit=sys_q,
                                             with_measurements=False)
    bell_u = cirq.unitary(measurement_bell)

    bell_u_test = create_measurement_circuit(system_qubits=system_qubits,
                                             sys_to_ancilla_map=qubit_map)
    bell_u_test = cirq.unitary(bell_u_test)
    assert np.isclose(abs(np.trace(bell_u_test.conj().T @ bell_u)), 4**n_qubits)