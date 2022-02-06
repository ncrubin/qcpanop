"""A set of modules for experimenting with the Kummer cone and Variety


{}^{2}D is the 2-particle operator.  When traced with another 2-particle
operator
"""
from itertools import product
import numpy as np
import scipy as sp
import openfermion as of
from openfermion.testing.hydrogen_integration_test import HydrogenIntegrationTest
import math
import fqe
from scipy.optimize import approx_fprime


def matrix_cofactor(matrix):
    return np.linalg.inv(matrix).T * np.linalg.det(matrix)


def matrix_cofactor_slow(matrix):
    C = np.zeros(matrix.shape)
    nrows, ncols = C.shape
    minor = np.zeros((nrows-1, ncols-1))
    for row in range(nrows):
        for col in range(ncols):
            minor[:row, :col] = matrix[:row, :col]
            minor[row:, :col] = matrix[row + 1:, :col]
            minor[:row, col:] = matrix[:row, col + 1:]
            minor[row:, col:] = matrix[row + 1:, col + 1:]
            C[row, col] = (-1)**(row+col) * np.linalg.det(minor)
    return C


def lift_from_two_to_n_particle(four_matrix, num_electrons):
    dim = four_matrix.shape[0]
    return of.get_number_preserving_sparse_operator(
             of.get_fermion_operator(
               of.InteractionOperator(0,  np.zeros((dim,)*2), four_matrix)
             ), dim, num_electrons
            ).toarray()


def delta_func(four_matrix_vec, nsorb, num_electrons):
    four_matrix = np.reshape(four_matrix_vec, (nsorb,)*4)
    return np.linalg.det(lift_from_two_to_n_particle(four_matrix, num_electrons))

def get_h2(bd):
    geometry = [['H', [0, 0, 0]],
                ['H', [0, 0, bd]]]
    molecule = of.MolecularData(geometry=geometry,
                                basis='sto-3',
                                charge=0,
                                multiplicity=1)
def main():
    dim = 2
    np.random.seed(19)
    ###################
    #
    #  Matrix adjugate
    #
    ###################
    A = np.random.randn(4 * dim**2).reshape((2 * dim, 2 * dim))
    A = A + A.T
    adA = matrix_cofactor(A)
    # print(adA)
    # print(matrix_cofactor_slow(A))
    # print(adA @ A / np.linalg.det(A))

    I2 = math.factorial(2) * of.wedge(np.identity(dim), np.identity(dim),
                                      (1, 1), (1, 1))
    for i, j, k, l in product(range(dim), repeat=4):
        if not np.isclose(I2[i, j, k, l], 0):
            print((i, j, l, k), I2[i, j, l, k])

    hinst = HydrogenIntegrationTest()
    hinst.setUp()
    molecule = hinst.molecule
    nelec = molecule.n_electrons
    nsorbs = molecule.n_orbitals * 2
    io_mol_ham = molecule.get_molecular_hamiltonian()
    red_ham = of.chem.make_reduced_hamiltonian(io_mol_ham, molecule.n_electrons)
    tpdm = molecule.fci_two_rdm

    k2 = red_ham.two_body_tensor
    nk2 = lift_from_two_to_n_particle(k2, molecule.n_electrons)
    w, v = np.linalg.eigh(nk2)
    assert np.isclose(w[0] + red_ham.constant, molecule.fci_energy)

    I_2 = of.wedge(np.identity(nsorbs), np.identity(nsorbs), (1, 1), (1, 1))
    # for i, j, k, l in product(range(nsorbs), repeat=4):
    #     if not np.isclose(I_2[i, j, k, l], 0):
    #         print((i, j, k, l), I_2[i, j, k, l])

    w, v = np.linalg.eigh(nk2)
    assert np.isclose(w[0] + molecule.nuclear_repulsion, molecule.fci_energy)
    gs_e, gs_wf = np.linalg.eigh(nk2)
    gs_e = gs_e[0]
    gs_wf = gs_wf[:, [0]]

    b2 = gs_e * I_2 / 2 - k2  # divide by 2 is from Bartlett paper.
    nb2 = lift_from_two_to_n_particle(b2, molecule.n_electrons)

    print((gs_e * np.identity(nk2.shape[0]) - nk2) @ gs_wf)
    print()
    print(nb2 @ gs_wf)

    print()

    # 2-RDM is defined as the derivative of |B^{N}| with respect to elements b2
    print(delta_func(b2.flatten(), nsorbs, nelec))
    tpdm_approx = approx_fprime(b2.flatten(), delta_func, 1.0E-4, nsorbs, nelec)
    tpdm_approx = tpdm_approx.reshape((nsorbs,)*4)
    # now imposing that the trace is the correct 2-RDM trace.
    trace_tpdm_approx = np.einsum('ijji', tpdm_approx)
    tpdm_approx /= trace_tpdm_approx / (nelec * (nelec - 1))
    print(np.einsum('ijji', tpdm_approx))

    tpdm_mat = tpdm_approx.transpose((0, 1, 3, 2)).reshape((nsorbs**2, nsorbs**2))
    print(np.linalg.norm(tpdm_mat - tpdm_mat.T))
    print(np.einsum('ijji', tpdm_approx))
    w, v = np.linalg.eigh(tpdm_mat)
    print(w)
    tpdm_mat_true = tpdm.transpose((0, 1, 3, 2)).reshape((nsorbs**2, nsorbs**2))
    w, v = np.linalg.eigh(tpdm_mat_true)
    print(w)
    print(np.einsum('ijkl,ijkl', tpdm, k2) + molecule.nuclear_repulsion, molecule.fci_energy)
    print(np.einsum('ijkl,ijkl', tpdm_approx, k2) + molecule.nuclear_repulsion, molecule.fci_energy)




if __name__ == '__main__':
    main()

