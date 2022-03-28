"""
Test the bell measurement for obtaining all qubit margiinals
"""
from qcpanop.qubit_measurement.qubit_marginal_ops import qubit_marginal_op_basis, get_qubit_marginals, get_qubit_marginal_expectations
from qcpanop.qubit_measurement.state_utils import get_random_state
from qcpanop.qubit_measurement.circuits import zeta_circuit, bell_measurement

import cirq
import numpy as np

from collections import defaultdict


def build_paulistring(qubit_list, pauli_type='X'):
    """Builds a PauliString from a list of qubits
    this is an entire funcntion because we might want to build
    different types of PauliStrings for different lists of qubits
    """
    bell_pair_dict = {}
    for qidx, qid in enumerate(qubit_list):
        if isinstance(pauli_type, list):
            bell_pair_dict[qid] = pauli_type[qidx]
        else:
            bell_pair_dict[qid] = pauli_type
    return cirq.PauliString(bell_pair_dict)


def main():
    np.random.seed(53)
    n_qubits = 4
    n_ancilla = n_qubits
    qubits = cirq.LineQubit.range(n_qubits + n_ancilla)
    qubit_map = dict(zip(qubits, range(n_qubits + n_ancilla)))
    ancilla_map = dict(zip(qubits[:n_qubits], qubits[n_qubits:]))

    # get a random state as a circuit.
    hadmard_circuit = cirq.Circuit([cirq.H.on(xx) for xx in qubits[:n_qubits]])
    random_circuit = cirq.testing.random_circuit(qubits[:n_qubits], n_moments=10, op_density=1,
                                                 gate_domain={cirq.rx(np.pi/3): 1,
                                                              cirq.CNOT: 2,
                                                              cirq.CZ: 2,
                                                              cirq.ISWAP: 2,
                                                              cirq.rz(np.pi/4): 1,
                                                              cirq.ry(np.pi/7): 1})
    random_circuit = hadmard_circuit + random_circuit

    # get the margials. This is ground truth
    marginals = get_qubit_marginals(
        random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 2)
    marginals_exp = get_qubit_marginal_expectations(random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 2)

    # build bell basis measurement circuit
    measurement_circuit = cirq.Circuit()
    measurement_bell = cirq.Circuit()
    for sys_q, anc_q in zip(qubits[:n_qubits], qubits[n_qubits:]):
        measurement_circuit += zeta_circuit(anc_q)
        measurement_bell += zeta_circuit(anc_q)
        measurement_bell += bell_measurement(ancilla_qubit=anc_q,
                                                system_qubit=sys_q,
                                                with_measuremet=False)
    final_state_bell = (random_circuit + measurement_bell).final_state_vector(qubit_order=qubits)
    final_state = (random_circuit + measurement_circuit).final_state_vector(qubit_order=qubits)
    print(random_circuit + measurement_bell)

    # build k-expectations for xx, xy, xz, yx, yy, yz, zx, zy, zz
    for key, val in marginals_exp.items():
        print(key)
        print(val)
        bell_pair_list = [(qidx, ancilla_map[qidx]) for qidx in key]
        bell_pair_list = [k for sub in bell_pair_list for k in sub]

        # now collect expectation values
        xx_paulistring = build_paulistring(bell_pair_list, 'X')
        xx_marginal_val = xx_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(xx_marginal_val * (np.sqrt(3)**2), val[cirq.X.on(key[0]) * cirq.X.on(key[1])])
        iz_paulistring = build_paulistring(bell_pair_list, ['I', 'Z', 'I', 'Z'])
        iz_bell_marginal_val = iz_paulistring.expectation_from_state_vector(final_state_bell, qubit_map=qubit_map).real
        print(iz_bell_marginal_val * np.sqrt(3)**2)
        print(xx_marginal_val * np.sqrt(3)**2)
        print(val[cirq.X.on(key[0]) * cirq.X.on(key[1])])


        yy_paulistring = build_paulistring(bell_pair_list, 'Y')
        yy_marginal_val = yy_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(yy_marginal_val * (np.sqrt(3)**2), val[cirq.Y.on(key[0]) * cirq.Y.on(key[1])])
        zz_paulistring = build_paulistring(bell_pair_list, ['Z', 'Z', 'Z', 'Z'])
        zz_bell_marginal_val = zz_paulistring.expectation_from_state_vector(final_state_bell, qubit_map=qubit_map).real
        assert np.isclose(zz_bell_marginal_val * (np.sqrt(3) ** 2), yy_marginal_val * (np.sqrt(3) ** 2))


        zz_paulistring = build_paulistring(bell_pair_list, 'Z')
        zz_marginal_val = zz_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(zz_marginal_val * (np.sqrt(3)**2), val[cirq.Z.on(key[0]) * cirq.Z.on(key[1])])
        zi_paulistring = build_paulistring(bell_pair_list, ['Z', 'I', 'Z', 'I'])
        zi_bell_marginal_val = zi_paulistring.expectation_from_state_vector(final_state_bell, qubit_map=qubit_map).real
        print(zi_bell_marginal_val * np.sqrt(3)**2)
        print(zz_marginal_val * np.sqrt(3)**2)
        print(val[cirq.Z.on(key[0]) * cirq.Z.on(key[1])])


        xy_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'Y', 'Y'])
        xy_marginal_val = xy_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(xy_marginal_val * (np.sqrt(3)**2), val[cirq.X.on(key[0]) * cirq.Y.on(key[1])])

        xz_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'Z', 'Z'])
        xz_marginal_val = xz_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(xz_marginal_val * (np.sqrt(3)**2), val[cirq.X.on(key[0]) * cirq.Z.on(key[1])])

        yx_paulistring = build_paulistring(bell_pair_list, ['Y', 'Y', 'X', 'X'])
        yx_marginal_val = yx_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(yx_marginal_val * (np.sqrt(3)**2), val[cirq.Y.on(key[0]) * cirq.X.on(key[1])])

        yz_paulistring = build_paulistring(bell_pair_list, ['Y', 'Y', 'Z', 'Z'])
        yz_marginal_val = yz_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(yz_marginal_val * (np.sqrt(3)**2), val[cirq.Y.on(key[0]) * cirq.Z.on(key[1])])

        xz_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'Z', 'Z'])
        xz_marginal_val = xz_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(xz_marginal_val * (np.sqrt(3)**2), val[cirq.X.on(key[0]) * cirq.Z.on(key[1])])

        ix_paulistring = build_paulistring(bell_pair_list, ['I', 'I', 'X', 'X'])
        ix_marginal_val = ix_paulistring.expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        assert np.isclose(ix_marginal_val * (np.sqrt(3)**1), val[cirq.X.on(key[1])])


    exit()
    # build expectation values
    z_expectations = defaultdict(dict)
    z_expectations_values = defaultdict(dict)
    xx_expectation_values = defaultdict(dict)
    for sys_q, anc_q in zip(qubits[:n_qubits], qubits[n_qubits:]):
        z_expectations[(sys_q, anc_q)]['ZsZa'] = cirq.PauliString({sys_q: 'Z', anc_q: 'Z'})
        z_expectations[(sys_q, anc_q)]['IsZa'] = cirq.PauliString({anc_q: 'Z'})
        z_expectations[(sys_q, anc_q)]['ZsIa'] = cirq.PauliString({sys_q: 'Z'})
        z_expectations_values[(sys_q, anc_q)]['ZsZa'] = z_expectations[(sys_q, anc_q)]['ZsZa'].expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        z_expectations_values[(sys_q, anc_q)]['IsZa'] = z_expectations[(sys_q, anc_q)]['IsZa'].expectation_from_state_vector(final_state, qubit_map=qubit_map).real
        z_expectations_values[(sys_q, anc_q)]['ZsIa'] = z_expectations[(sys_q, anc_q)]['ZsIa'].expectation_from_state_vector(final_state, qubit_map=qubit_map).real

        xx_expectation_values[(sys_q, anc_q)]['XsXa']


    for key, val in z_expectations_values.items():
        print(val)




if __name__ == "__main__":
    main()