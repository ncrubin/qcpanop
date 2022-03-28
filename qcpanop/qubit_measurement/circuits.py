import numpy as np
import cirq


def zeta_circuit(ancilla_qubit) -> cirq.Circuit:
    """
    Construct ancilla preparation circuit.  Expectationn of this state
    is completely symmetric in sigma-x, sigma-y, and sigma-z

    :param ancilla_qubit:
    :return:
    """
    theta = np.arccos(1/np.sqrt(3))
    phi = 3 * np.pi / 4
    zeta_state = cirq.Circuit([cirq.rx(theta).on(ancilla_qubit),
                               cirq.rz(phi).on(ancilla_qubit)])
    return zeta_state


def bell_measurement(ancilla_qubit, system_qubit,
                     with_measuremet=True) -> cirq.Circuit:
    """
    Measurement gadget to measure a pair of qubits in the Bell basis

    :param ancilla_qubit: ancilla qubit - control + H bit
    :param system_qubit: Not bit
    :return:
    """
    if with_measuremet:
        measure_circuit = cirq.Circuit([cirq.CNOT.on(ancilla_qubit, system_qubit),
                                        cirq.H.on(ancilla_qubit),
                                        cirq.measure([ancilla_qubit, system_qubit],
                                                     key=repr(system_qubit))])
    else:
        measure_circuit = cirq.Circuit([cirq.CNOT.on(ancilla_qubit, system_qubit),
                                        cirq.H.on(ancilla_qubit)])

    return measure_circuit
