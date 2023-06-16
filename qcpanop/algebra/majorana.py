import itertools
import numpy as np
import scipy as sp
from scipy.linalg import schur
from scipy.linalg import block_diag

import matplotlib.pyplot as plt

import openfermion as of


def omega_transform(dim: int):
    """
    :param dim: number fermionic modes
    """
    omega_row1 = np.hstack((np.eye(dim), np.eye(dim)))
    omega_row2 = np.hstack((1j * np.eye(dim), -1j * np.eye(dim)))
    return np.vstack((omega_row1, omega_row2)) / np.sqrt(2)

def xp_to_xx_majorana_order(dim, verify=False):
    """
    :param dim: Number of Majorana modes -- should be 2 * |fermionic_modes|
    """
    if verify:
        assert dim % 2 == 0
    f_transform = np.eye(dim)
    indices = np.arange(dim)
    front_half_indices = indices[:dim//2]
    back_half_indices = indices[dim//2:]
    new_indices = [item for sublist in zip(front_half_indices, back_half_indices) for item in sublist]
    f_transform = f_transform[:, new_indices]
    return f_transform


def general_quadratic_hamiltonian_ladder_op_form(A, B):
    """
    :param A: hermitian matrix for A_{ij}a_{i}^ a_{j}
    :param B: skew symmetric matrix for B_{ij}a_{i}a_{j} 
    """
    quad_h_row1 = np.hstack((-A.conj(), B))
    quad_h_row2 = np.hstack((-B.conj(), A))
    return np.vstack((quad_h_row1, quad_h_row2))

def build_correct_majorana_order(block_mat):
    dim = block_mat.shape[0] // 2
    smats = []
    for ii in range(dim):
        if block_mat[2 * ii, 2 * ii + 1] <= 0:
            smats.append(np.array([[0, 1], [1, 0]]))
        else:
            smats.append(np.array([[1, 0], [0, 1]]))
    return block_diag(*smats)

def block_diagonalize_skew_symmetry(mat, verify=False):
    """
    Block diagonalize a skew-symmetric matrix

    :param mat: skew symmetric matrix that must be even dimension
    :param verify: optional value to check if skew-symmetric. Default = False 
    :returns: T, Z. T is block diagonal matrix with positive eigenvalue on the upper triangle.
              Z is the orthogonal rotation that performs this operation
    """
    if verify:
        if not np.allclose(mat.T, -mat):
            raise TypeError("Input is not a skew symmetric real matrix")
        if mat.shape[0] % 2 == 1:
            raise TypeError("skew symmetric matrix must have even size")

    Ttilde, Ztilde = schur(mat.real, output='real')
    smat = build_correct_majorana_order(Ttilde)
    Z = Ztilde @ smat
    T = Z.T @ mat.real @ Z
    return T, Z

def tutorial_on_dirac_majorana_diagonalization():
    dim = 6
    omega = omega_transform(dim)
    assert np.allclose(omega.conj().T @ omega, np.eye(2 * dim))

    A = np.random.randn(dim**2).reshape((dim, dim)) + 1j * np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.conj().T
    B = np.random.randn(dim**2).reshape((dim, dim)) 
    B = B - B.conj().T

    np.fill_diagonal(B, 0)
    assert np.allclose(B.T, -B)
    QH = general_quadratic_hamiltonian_ladder_op_form(A, B)
    assert np.allclose(QH, QH.conj().T) # the hamiltonian is hermitan

    # the -1j cancels out the 1j in the definition ih = Omega H Omega.H 
    # h is real skew symmetric matrix
    majorana_H = -1j * (omega @ QH @ omega.conj().T)
    assert np.allclose(majorana_H[:dim, :dim], (A + B).imag)
    assert np.allclose(majorana_H[dim:, dim:], (A - B).imag)
    assert np.allclose(majorana_H[:dim, dim:], (A + B).real)
    assert np.allclose(majorana_H[dim:, :dim], (B - A).real)
    assert np.allclose(majorana_H.imag, 0)
    assert np.allclose(majorana_H.T, -majorana_H)

    # Diagonal form dirac operators
    w, v = np.linalg.eigh(QH)
    print("eigenvalues of smmetry hermitian Hamiltonian")
    print(w)

    Ttilde, Ztilde = schur(majorana_H.real, output='real')  # this puts negative abs-val largest first then real
    assert np.allclose(Ztilde.imag, 0)
    assert np.allclose(Ztilde.T @ Ztilde, np.eye(2 * dim))
    smat = build_correct_majorana_order(Ttilde)
    Z = Ztilde @ smat
    T = Z.T @ majorana_H.real @ Z
    assert np.allclose(Z.imag, 0)
    assert np.allclose(Z.T @ Z, np.eye(2 * dim))
    plt.imshow(T)
    plt.show()
    f_transform = xp_to_xx_majorana_order(2 * dim)


    dirac_order_diagonalizer_from_majorana = omega.conj().T @ Z @ f_transform.T @ omega
    assert np.allclose(dirac_order_diagonalizer_from_majorana.conj().T @ dirac_order_diagonalizer_from_majorana,
                       np.eye(2 * dim))
    QH_d = dirac_order_diagonalizer_from_majorana.conj().T @ QH @ dirac_order_diagonalizer_from_majorana
    print(np.diag(QH_d).real)
    plt.imshow(QH_d.real)
    plt.show() 


def gaussian_state_test():
    # single mode gaussian state tutorial
    epsilon = np.pi/3
    beta = 1.75
    op = of.FermionOperator(((0, 1), (0, 0))) - of.FermionOperator(((0, 0), (0, 1))) 
    op *= epsilon
    hmat = of.get_sparse_operator(op).todense()
    exp_op = sp.linalg.expm(-beta * hmat)
    print(exp_op)
    normalization_constant = np.trace(exp_op)
    rho = exp_op / normalization_constant
    assert np.isclose(np.trace(rho), 1)
    print(np.cosh(beta * epsilon) + np.sinh(beta * epsilon))
    print(np.cosh(beta * epsilon) - np.sinh(beta * epsilon))
    print("Z = ", normalization_constant, 2 * np.cosh(beta * epsilon))
    assert np.isclose(normalization_constant, 2 * np.cosh(beta * epsilon))
    print(rho)
    print((np.cosh(beta * epsilon) + np.sinh(beta * epsilon)) / normalization_constant)
    print((np.cosh(beta * epsilon) - np.sinh(beta * epsilon)) / normalization_constant)


    op = of.FermionOperator(((0, 1), (0, 0)), coefficient=epsilon * -beta)
    number_op = of.get_sparse_operator(op).todense()
    exp_number_op = sp.linalg.expm(number_op)
    print(exp_number_op)
    print(1 + np.exp(epsilon * -beta))

def get_opdm(rho, dim):
    lower_ops = [of.get_sparse_operator(of.FermionOperator(((ii, 0))), n_qubits=dim) for ii in range(dim)]
    raise_ops = [xx.conj().T for xx in lower_ops]

    opdm = np.zeros((dim, dim), dtype=np.complex128)
    for i, j in itertools.product(range(dim), repeat=2):
        opdm[i, j] = np.trace(rho @ raise_ops[i] @ lower_ops[j])
    return opdm

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    dim = 4
    gamma = np.random.randn(dim**2).reshape((dim, dim)) + 1j * np.random.randn(dim**2).reshape((dim, dim))
    gamma = gamma + gamma.conj().T
    assert np.allclose(gamma, gamma.conj().T)

    w, v = np.linalg.eigh(gamma)
    normalization_constant = np.product(1 + np.exp(w))
    assert np.allclose(v.conj().T @ gamma @ v, np.diag(w))

    import cirq
    from openfermion.circuits.primitives import optimal_givens_decomposition

    givens_circuit = cirq.Circuit(optimal_givens_decomposition(cirq.LineQubit.range(dim), v.copy()))
    givens_unitary = cirq.unitary(givens_circuit)

    rho = of.FermionOperator() # un-normalized
    for i, j in itertools.product(range(dim), repeat=2):
        rho += of.FermionOperator(((i, 1), (j, 0)), coefficient=gamma[i, j])
    rho_mat = of.get_sparse_operator(rho).todense()
    rho_mat = sp.linalg.expm(rho_mat) / normalization_constant
    assert np.isclose(rho_mat.trace(), 1)
    assert np.allclose(rho_mat, rho_mat.conj().T)
    true_opdm = get_opdm(rho_mat, dim)

    nop = of.get_sparse_operator(of.FermionOperator(((0, 1), (0, 0)))).todense()
    exp_lambda_n = sp.linalg.expm((np.pi/3) * nop)
    assert np.allclose(exp_lambda_n, np.eye(2) + (np.exp(np.pi/3) - 1) * nop)
    print(np.trace(exp_lambda_n @ nop), np.exp(np.pi/3))
    print(np.exp(np.pi/3) / (1 + np.exp(np.pi/3)))
    diagonal_op = of.FermionOperator()
    for ii in range(dim):
        diagonal_op += of.FermionOperator(((ii, 1), (ii, 0)), coefficient=w[ii])
    exp_diagonal_op = sp.linalg.expm(of.get_sparse_operator(diagonal_op).todense())
    assert np.isclose(np.trace(exp_diagonal_op), normalization_constant)
    rho_diag = exp_diagonal_op / normalization_constant
    diag_opdm = get_opdm(rho_diag, dim)
    print(diag_opdm)
    test_diag_opdm = np.diag([np.exp(w_i) / (1 + np.exp(w_i)) for w_i in w])
    print(test_diag_opdm)
    assert np.allclose(diag_opdm, test_diag_opdm)

    test_lambda = -np.log(np.reciprocal([np.exp(w_i) / (1 + np.exp(w_i)) for w_i in w]) - 1)
    assert np.allclose(test_lambda, w)

    test_gamma = v @ np.diag(test_lambda) @ v.conj().T
    assert np.allclose(test_gamma, gamma)

    topdm = np.einsum("ik,jk,kk", v.conj(), v, test_diag_opdm)
    assert np.allclose(topdm, true_opdm)
    print(topdm)
    print(true_opdm)
