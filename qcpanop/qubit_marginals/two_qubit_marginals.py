"""
Explore the adjoint representation
"""
from collections import defaultdict
from itertools import combinations, product
import numpy as np
import cirq
from cirq.ops.linear_combinations import _pauli_string_from_unit
import openfermion as of

from qcpanop.qubit_measurement.qubit_marginal_ops import qubit_marginal_op_basis

from qcpanop.qubit_marginals.algebra import OperatorBasis
from qcpanop.qubit_marginals.utils import n_qubit_pauli_basis

def get_operator_basis(marginal_rank, qubits):
    marginal_op_basis = qubit_marginal_op_basis(marginal_rank, qubits)
    operator_basis = []
    for q_pair, op_terms in marginal_op_basis.items():
        for term in op_terms:
            operator_basis.append(term)

    operator_basis = sorted(list(set(operator_basis)), key=repr)  # define some consist ordering.
    ob_dict = dict(zip(operator_basis, range(len(operator_basis))))
    return operator_basis, ob_dict


def paulisum_to_paulistring(paulisum_term: cirq.PauliSum) -> cirq.PauliString:
    """
    Convert a PauliSum to Paulistring. this is going down in the abstraction
    hiearchy

    :param paulisum_term:
    :return:
    """
    if not isinstance(paulisum_term, cirq.PauliSum):
        raise ValueError("wrong input type")
    if len(paulisum_term) > 1:
        raise ValueError("PauliSum is not singleton")

    terms = [
        (_pauli_string_from_unit(v), paulisum_term._linear_dict[v]) for v in
        paulisum_term._linear_dict.keys()
    ]
    if len(terms) == 0:
        return cirq.PauliString((), coefficient=0)
    else:
        return terms[0][0] * terms[0][1]


def get_naked_pauliterm(pauliterm: cirq.PauliString) -> cirq.PauliString:
    """
    Get a PauliString without with the coefficient set to 1

    :param pauliterm:
    :return:
    """
    if not isinstance(pauliterm, cirq.PauliString):
        raise ValueError("wrong input type")
    naked_term = cirq.PauliString(coefficient=1.,
                                  qubit_pauli_map=pauliterm._qubit_pauli_map)
    return naked_term

def main():
    from scipy.special import comb
    n_qubits = 6
    qubits = cirq.LineQubit.range(n_qubits)
    rank_1_operater_basis, r1_ob_dict = get_operator_basis(1, qubits)  # n choose 2  4x4 density matricse O(n^2)
    rank_2_operater_basis, r2_ob_dict = get_operator_basis(2, qubits)  # n choose 2  4x4 density matricse O(n^2)
    rank_3_operater_basis, r3_ob_dict = get_operator_basis(3, qubits)  # n choose 2  4x4 density matricse O(n^2)
    rank_4_operater_basis, r4_ob_dict = get_operator_basis(4, qubits)  # n choose 4 16x16 density matrices O(n^4)
    print(n_qubits * 3 + 1, len(rank_1_operater_basis), sum([comb(n_qubits, xx) * (3**xx) for xx in range(1 + 1)]))
    print(int(comb(n_qubits, 2) * 9 + n_qubits * 3 + 1), len(rank_2_operater_basis), sum([comb(n_qubits, xx) * (3**xx) for xx in range(2 + 1)]))
    print(int(comb(n_qubits, 3) * (3**3) + comb(n_qubits, 2) * 9 + n_qubits * 3 + 1), len(rank_3_operater_basis), sum([comb(n_qubits, xx) * (3**xx) for xx in range(3 + 1)]))
    print(int(comb(n_qubits, 4) * (3**4) + comb(n_qubits, 3) * (3**3) + comb(n_qubits, 2) * 9 + n_qubits * 3 + 1), len(rank_4_operater_basis), sum([comb(n_qubits, xx) * (3**xx) for xx in range(4 + 1)]))

    for term1, idx1 in r2_ob_dict.items():
        for term2, idx2 in r3_ob_dict.items():
            comm_t1_t2 = term1 * term2 - term2 * term1  # returns PauliSum --not PauliString
            assert len(comm_t1_t2) == 0 or len(comm_t1_t2) == 1  # make sure a single algebra element is formed
            comm_t1_t2 = paulisum_to_paulistring(comm_t1_t2)
            naked_term = get_naked_pauliterm(comm_t1_t2)
            print(term1, term2)
            print(term1, term2, comm_t1_t2, naked_term, comm_t1_t2 == naked_term, len(naked_term))
            print()
            assert len(naked_term) <= 3






if __name__ == "__main__":
    main()