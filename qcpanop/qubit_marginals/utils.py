import numpy as np

from qcpanop.qubit_marginals.algebra import OperatorBasis

I = 'I'
X = 'X'
Y = 'Y'
Z = 'Z'

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1.j], [1.j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

pauli_label_ops = [(I, np.eye(2)), (X, sigma_x), (Y, sigma_y), (Z, sigma_z)]


def pauli_basis_measurements(qubit):
    """
    Generates the Programs required to measure the expectation values of the pauli operators.

    :param qubit: Required argument (so that the caller has a reference).
    :return:
    """
    pauli_label_meas_progs = [Program(), Program(RY(-np.pi/2, qubit)), Program(RX(-np.pi/2, qubit)), Program()]
    return pauli_label_meas_progs


PAULI_BASIS = OperatorBasis(pauli_label_ops)


def n_qubit_pauli_basis(n):
    """
    Construct the tensor product operator basis of `n` PAULI_BASIS's.

    :param int n: The number of qubits.
    :return: The product Pauli operator basis of `n` qubits
    :rtype: OperatorBasis
    """
    if n >= 1:
        return PAULI_BASIS ** n
    else:
        raise ValueError("n = {} should be at least 1.".format(n))
