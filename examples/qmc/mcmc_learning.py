from itertools import product
import numpy as np
from pyscf import gto, scf, fci
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

from qcpanop.ci.didacticgfmc import DidacticGFMC, print_wfn

def mcmc_example_for_learning():
    """
    Build simple MCMC to sample ground state of matrix A.
    """
    np.set_printoptions(linewidth=500)
    np.random.seed(10)
    dim = 5
    A = np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.T
    A *= np.sign(A)  # get positive matrix
    assert np.allclose(A, A.T)
    sigma = 10
    sigmaI = np.eye(dim) * sigma
    # we want to define a stochastic matrix from A
    # A_{ij} = w_{ij}P_{ij} where P_{ij} is stochastic
    P = np.zeros((dim, dim))
    for i, j in product(range(dim), repeat=2):
        P[i, j] = A[i, j] / np.sum(A[:, j])
    # print(np.einsum('ij->j', P))
    # print(P)
    # print()

    w, v = np.linalg.eig(P)
    idx_sorted = np.argsort(w)
    w = w[idx_sorted]
    v = v[:, idx_sorted]
    mu = np.sign(v[0, -1]) * v[:, [-1]]
    # for i, j in product(range(dim),repeat=2):
    #     print(mu[i] * P[i, j], mu[j] * P[j, i])

    weight = np.zeros((dim, dim))
    for i, j in product(range(dim), repeat=2):
        weight[i, j] = np.sum(A[:, j])

    assert np.allclose(A, weight * P)

    w, v = np.linalg.eigh(A)
    # print(w)
    # print()
    # print(v)

    residual = 0
    x0 = np.ones(dim).reshape((-1, 1))
    x0 /= dim  # np.linalg.norm(x0)  # need to make sum_{i}w_{i} = 1 for init state
    x = x0.copy()
    x_old = x0.copy()
    iter = 0
    iter_max = 20
    while iter < iter_max and residual < (1 - 1.0E-5):
        x = A @ x
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        # print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1
    # print(x)



    # print("probability and stochastic matrices")
    # print(P)
    norm_val = sum(x0)
    assert np.isclose(sum(P @ x0 / norm_val), 1)

    # print()
    # print(x0)
    # print(P @ P @ P @ x0)
    # print(np.einsum('ij,jk,kl,l->i', P, P, P, x0.flatten()))

    # so we need o4 iterations to converge so let's imagine we have
    # configurations i_{1}, i_{2}, i_{3}, i_{4} where each i can take on dim values
    configurations = list(product(range(dim), repeat=4))
    probabilties = {}
    weight_expectation = {}

    final_weight_i = np.zeros(dim)
    for config in configurations:
        i1, i2, i3, i4 = config
        final_weight_i += (weight[:, i4] * P[:, i4]) * (weight[i4, i3] * P[i4, i3] ) \
            * (weight[i3, i2] * P[i3, i2]) * (weight[i2, i1] * P[i2, i1]) * x0[i1]
        probabilties[config] = P[:, i4] * P[i4, i3] * P[i3, i2] * P[i2, i1] * x0[i1]
        weight_expectation[config] = weight[:, i4] * weight[i4, i3] * weight[i3, i2] * weight[i2, i1]

    final_weight_i = final_weight_i.reshape((-1, 1))
    print("Summed final weight")
    print(final_weight_i)
    print()
    final_vec = A @ A @ A @ A @ x0
    print("True final weight")
    print(final_vec)
    print()

    i_0_prob = np.zeros(dim)
    for config in configurations:
        i_0_prob += probabilties[config]
    print("Summed probabilities")
    print(i_0_prob.reshape((-1, 1)))
    print(np.sum(i_0_prob))
    print()

    print("Final Weight vector from expectation value")
    final_weight_i = np.zeros(dim)
    for config in configurations:
        final_weight_i += probabilties[config] * weight_expectation[config]
    final_weight_i = final_weight_i.reshape((-1, 1))
    print(final_weight_i)
    print()


    def b_x(ii):
        return np.sum(A[:, ii])


    # single walker evolution and sample
    num_samples = 100_000
    possible_walker_indices = list(range(dim))
    sampled_final_output = np.zeros(dim)
    sampled_ritz_coeff = 0
    walker_weighted_expectation = 0
    walker_weight_expectation = 0
    for _ in range(num_samples):
        # randomly sample an initial walker configuration
        walker_position = np.random.randint(0, dim)
        walker_weight = 1

        for step in range(4):  # because we do 4 mat-vec operations
            # sample new position from column space
            new_walker_position = np.random.choice(possible_walker_indices, p=P[:, walker_position])

            # update weight and walker
            walker_weight = weight[new_walker_position, walker_position] * walker_weight
            walker_position = new_walker_position
        sampled_final_output[walker_position] += walker_weight

        walker_weighted_expectation += walker_weight * b_x(walker_position)
        walker_weight_expectation += walker_weight

    walker_weighted_expectation /= num_samples
    walker_weight_expectation /= num_samples

    print("Stochastically sampled final vector")
    sampled_final_output /= num_samples
    sampled_final_output = sampled_final_output.reshape((-1, 1))
    print("diff :", np.linalg.norm(sampled_final_output - final_weight_i))
    # the above single walker version is trivially extended to many walkers
    # such that we don't loop over samples we just propagate foward the entire
    # population of walkers+weights.  At the end we then sum up all the
    # energies and then divide by the population.
    print(sampled_final_output / np.linalg.norm(sampled_final_output))

    w, v = np.linalg.eigh(A)
    print("True Eigenspectrum of A")
    print(w)
    print("<wb>/<w>")
    print(walker_weighted_expectation / walker_weight_expectation)

    print(v)
    # if I want the expectation value of A then I can estimate it from
    # samples of my ground state.  We are sampling output of the power method
    # which is proportional to the ground state where the proportionality  is lambda

    walker_idx = np.random.randint(0, dim, num_samples)
    print(walker_idx)


def big_matrix():
    nqubits = 6
    from s0experiment.hamiltonians.rg_hamiltonian import RGDOCI
    rgham = RGDOCI(8, nqubits)
    hamiltonian, constant = rgham.get_qubit_hamiltonian()
    hamiltonian += of.QubitOperator((), constant)


    gfmc_inst = DidacticGFMC(hamiltonian, 10_000)
    A = gfmc_inst.sparse_ham.toarray().real
    print(A)
    assert np.alltrue(A>=0)

    w, v = np.linalg.eigh(A)
    print(w)
    print_wfn(v[:, [-1]], nqubits)

    sigma = 10
    dim = 2**nqubits
    sigmaI = np.eye(dim) * sigma
    residual = np.inf
    phi0 = np.random.randn(dim).T
    x = phi0.copy()
    x_old = phi0.copy()
    x_future = phi0.copy()
    x_future = v[:, [-1]]
    for ii in range(2**nqubits):
        if not np.isclose(x_future[ii], 0):
            x_future[ii, 0] += 0.01 * np.random.randn()
    iter = 0
    iter_max = 500

    while residual > 1.0E-4 and iter < iter_max:
        # instead of applying (A - sigma I)^{-1} to x_{i} we obtain
        # x_{i+1} by solving (A - sigmaI)x_{i+1} = x_{i} --i.e. Ax = b
        # x = (A - sigmaI) @ x
        x = A  @ x
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        if iter % 50 == 0:
            print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1
    print("correct sign is lam/x[0,0]", np.sign(lam/x[0]))
    print("Error :", (x.T @ A @ x / (x.T @ x)) - w[-1])



if __name__ == "__main__":
    # mcmc_example_for_learning()
    big_matrix()

