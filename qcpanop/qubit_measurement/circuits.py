from typing import Dict, List
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
                     with_measurements=True) -> cirq.Circuit:
    """
    Measurement gadget to measure a pair of qubits in the Bell basis

    :param ancilla_qubit: ancilla qubit - control + H bit
    :param system_qubit: Not bit
    :return:
    """
    if with_measurements:
        measure_circuit = cirq.Circuit([cirq.CNOT.on(ancilla_qubit, system_qubit),
                                        cirq.H.on(ancilla_qubit),
                                        cirq.measure([ancilla_qubit, system_qubit],
                                                     key=repr(system_qubit))])
        # measure_circuit = cirq.Circuit([cirq.H.on(ancilla_qubit),
        #                                 cirq.CZ.on(ancilla_qubit, system_qubit),
        #                                 cirq.measure([ancilla_qubit, system_qubit],
        #                                              key=repr(system_qubit) + "_" + repr(ancilla_qubit))])
    else:
        # measure_circuit = cirq.Circuit([cirq.CNOT.on(ancilla_qubit, system_qubit),
        #                                 cirq.H.on(ancilla_qubit)])
        measure_circuit = cirq.Circuit([cirq.H.on(system_qubit),
                                        cirq.CZ.on(ancilla_qubit, system_qubit),
                                        cirq.H.on(ancilla_qubit),
                                        cirq.H.on(system_qubit)
                                        ])


    return measure_circuit


def create_measurement_circuit(system_qubits: List[cirq.Qid],
                               sys_to_ancilla_map: Dict[
                                   cirq.Qid, cirq.Qid],
                               compile=True) -> cirq.Circuit:
    """Create the right aligned measurement circuit that performs the bell
    measurement

    :param system_qubits: List of system qubits
    :param sys_to_ancilla_map: Dictionary key is system Qid value is ancilla Qid"""
    measurement_circuit = cirq.Circuit()
    for sys_q in system_qubits:
        anc_q = sys_to_ancilla_map[sys_q]
        measurement_circuit += zeta_circuit(ancilla_qubit=anc_q)
        measurement_circuit += bell_measurement(ancilla_qubit=anc_q,
                                                system_qubit=sys_q,
                                                  with_measurements=False)
    if compile:
        measurement_circuit = cirq.transformers.merge_single_qubit_moments_to_phxz(measurement_circuit)
        measurement_circuit = cirq.transformers.align_right(measurement_circuit)
        measurement_circuit = cirq.transformers.merge_single_qubit_moments_to_phxz(measurement_circuit)
        measurement_circuit = cirq.transformers.align_right(measurement_circuit)
    return measurement_circuit


if __name__ == "__main__":
    qubits = cirq.LineQubit.range(12)
    system_qubits = qubits[:6]
    n_qubits = len(system_qubits)
    ancilla_qubits = qubits[6:]
    sys_to_anc_map = dict(zip(system_qubits, ancilla_qubits))
    mcircuit = create_measurement_circuit(system_qubits, sys_to_ancilla_map=sys_to_anc_map)
    print(mcircuit)

