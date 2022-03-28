from .qubit_marginal_ops import (qubit_marginal_op_basis, get_qubit_marginals,
                                 get_qubit_marginals_cirq,
                                 get_qubit_marginal_expectations, )
from .state_utils import get_random_state

import numpy as np
import cirq


def test_qubit_marginal_expectations(verbose=False):
    np.random.seed(53)
    n_qubits = 6
    marginal_rank = 2
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    state = get_random_state(n_qubits)
    qubit_marginal_basis = qubit_marginal_op_basis(marginal_rank=marginal_rank, qubits=qubits)
    marginal_dict = get_qubit_marginal_expectations(state, qubits, marginal_rank)
    for key, val in qubit_marginal_basis.items():
        for pterm in val:
            if verbose:
                print(pterm,
                      pterm.expectation_from_state_vector(state, qubit_map=qubit_map).real,
                      marginal_dict[key][pterm])
            assert np.isclose(pterm.expectation_from_state_vector(state, qubit_map=qubit_map).real,
                              marginal_dict[key][pterm])


def test_qubit_marginal_density_matrices(verbose=False):
    np.random.seed(53)
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    state = get_random_state(n_qubits)
    rho_full = cirq.density_matrix_from_state_vector(state_vector=state)
    qubit_marginal_basis = qubit_marginal_op_basis(2, qubits)

    for key, val in qubit_marginal_basis.items():
        test_rho = np.zeros((2**len(key), 2**len(key)), dtype=np.complex128)
        rho_true = cirq.density_matrix_from_state_vector(state_vector=state, indices=[kk._x for kk in key])
        for pterm in val:
            pauli_op_mat = cirq.unitary(cirq.Circuit([cirq.I(xx) for xx in qubits] + [pterm]))
            test_pauli_expect = (state.conj().T @ pauli_op_mat @ state).real
            if verbose:
                print(pterm, pterm.expectation_from_state_vector(state, qubit_map=qubit_map).real,
                      test_pauli_expect)

            pauli_op_mat_marginal = cirq.unitary(cirq.Circuit([cirq.I(xx) for xx in key] + [pterm]))
            test_rho += test_pauli_expect * pauli_op_mat_marginal / 2**len(key)

        # test if marginal is equivalent to traced out marginal
        assert np.allclose(test_rho, rho_true)

        # check if properties are correct
        assert np.isclose(test_rho.trace(), 1.)
        assert np.allclose(test_rho, test_rho.conj().T)
        w, v = np.linalg.eigh(test_rho)
        assert np.isclose(len(np.where(w < -1.0E-13)[0]), 0)


def test_acquire_marginals():
    np.random.seed(53)
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    state = get_random_state(n_qubits)
    marginal_dict_true = get_qubit_marginals_cirq(state, qubits, 2)
    marginal_dict_test = get_qubit_marginals(state, qubits, 2)
    for key, val in marginal_dict_true.items():
        assert np.allclose(val, marginal_dict_test[key])

    n_qubits = 6
    qubits = cirq.LineQubit.range(n_qubits)
    qubit_map = dict(zip(qubits, range(n_qubits)))
    state = get_random_state(n_qubits)
    marginal_dict_true = get_qubit_marginals_cirq(state, qubits, 3)
    marginal_dict_test = get_qubit_marginals(state, qubits, 3)
    for key, val in marginal_dict_true.items():
        assert np.allclose(val, marginal_dict_test[key])

    marginal_dict_true = get_qubit_marginals_cirq(state, qubits, 1)
    marginal_dict_test = get_qubit_marginals(state, qubits, 1)
    for key, val in marginal_dict_true.items():
        assert np.allclose(val, marginal_dict_test[key])


