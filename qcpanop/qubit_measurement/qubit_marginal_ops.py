"""
Implementation of Zhang's Bell measurement to obtain
all k-qubit RDM information
"""
from itertools import product
from itertools import combinations
from collections import defaultdict

import numpy as np
import cirq


def qubit_marginal_op_basis(marginal_rank, qubits):
    """
    Generate operator basis for the k-qubit marginals

    :param marginal_rank: 1-qubit marginals, 2-qubit marginals etc.
    :param qubits: list of Qid objects
    :return:
    """
    qubit_type = {0: cirq.I, 1: 'X', 2: 'Y', 3: 'Z'}
    qubit_sets = defaultdict(list)
    n_qubits = len(qubits)
    for qs in combinations(range(n_qubits), marginal_rank):
        qs_sorted = tuple(sorted([qubits[qq] for qq in qs]))
        for pauli_term_per_qubit in product(range(4), repeat=len(qs)):
            paulistring = dict([(qubits[qs[qidx]], qubit_type[pauli_type]) for qidx, pauli_type in enumerate(pauli_term_per_qubit)])
            qubit_sets[qs_sorted].append(cirq.PauliString(paulistring))

    return qubit_sets


def get_qubit_marginals_cirq(state, qubits, qubit_marginal_rank):
    """
    Calculate the qubit marginals for a state using cirq's marginalization
    routines

    :param state:
    :param n_qubits:
    :return:
    """
    qubit_marginal_basis = qubit_marginal_op_basis(qubit_marginal_rank, qubits)
    marginal_dict = {}
    for key, val in qubit_marginal_basis.items():
        rho_true = cirq.density_matrix_from_state_vector(state_vector=state,
                                                         indices=[kk._x for kk in
                                                                  key])
        marginal_dict[key] = rho_true
    return marginal_dict


def get_qubit_marginals(state, qubits, qubit_marginal_rank):
    """
    Calculate the qubit marginals for a state

    :param state:
    :param n_qubits:
    :return:
    """
    qubit_marginal_basis = qubit_marginal_op_basis(qubit_marginal_rank, qubits)
    marginal_dict = {}
    for key, val in qubit_marginal_basis.items():
        rho = np.zeros((2**len(key), 2**len(key)), dtype=np.complex128)
        for pterm in val:
            pauli_op_mat = cirq.unitary(cirq.Circuit([cirq.I(xx) for xx in qubits] + [pterm]))
            test_pauli_expect = (state.conj().T @ pauli_op_mat @ state).real
            pauli_op_mat_marginal = cirq.unitary(cirq.Circuit([cirq.I(xx) for xx in key] + [pterm]))
            rho += test_pauli_expect * pauli_op_mat_marginal / 2**len(key)
        marginal_dict[key] = rho
    return marginal_dict


def get_qubit_marginal_expectations(state, qubits, qubit_marginal_rank):
    """
    Calculate the qubit marginals for a state

    :param state:
    :param n_qubits:
    :return:
    """
    qubit_map = dict(zip(qubits, range(len(qubits))))
    qubit_marginal_basis = qubit_marginal_op_basis(qubit_marginal_rank, qubits)
    marginal_dict = defaultdict(dict)
    for key, val in qubit_marginal_basis.items():
        for pterm in val:
            marginal_dict[key][pterm] = pterm.expectation_from_state_vector(state, qubit_map=qubit_map).real
    return marginal_dict
