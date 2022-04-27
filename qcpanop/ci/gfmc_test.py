from itertools import product
import numpy as np
from pyscf import gto, scf, fci
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

from qcpanop.ci.didacticgfmc import DidacticGFMC, PsiTrial

def power_method_example1():
    """Example for NCR to learn power method"""
    dim = 10
    A = np.array([[2, -12], [1, -5]], dtype=float)
    A = np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.T
    x0 = np.random.randn(dim).T
    w, v = np.linalg.eigh(A)
    print(w)

    residual = np.inf
    x = x0.copy()
    x_old = x0.copy()
    iter = 0
    iter_max = 100
    while residual > 1.0E-4 and iter < iter_max:
        x = A @ x
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        if iter % 10 == 0:
            print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1
    print("correct sign is lam/x[0,0]", np.sign(lam/x[0]))



def shifted_inverse_example():
    """Shifted inverse. Find eigenvalue closes to sigma (A - sigmaI)^{-1} power
    iteration. Example for learning"""
    np.random.seed(10)
    dim = 10
    A = np.array([[2, -12], [1, -5]], dtype=float)
    A = np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.T
    x0 = np.random.randn(dim).T
    w, v = np.linalg.eigh(A)
    print(w)
    print(np.reciprocal(w))
    print(1/w)
    sigma = -4
    sigmaI = sigma * np.eye(dim)

    residual = np.inf
    x = x0.copy()
    x_old = x0.copy()
    iter = 0
    iter_max = 100

    while residual > 1.0E-4 and iter < iter_max:
        # instead of applying (A - sigma I)^{-1} to x_{i} we obtain
        # x_{i+1} by solving (A - sigmaI)x_{i+1} = x_{i} --i.e. Ax = b
        x = np.linalg.solve(A - sigmaI, x)
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        if iter % 10 == 0:
            print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1
    print("correct sign is lam/x[0,0]", np.sign(lam/x[0]))


def shifted_lambda_example():
    """Shifted such that power method gets ground state--i.e. lambda farthest from sigma"""
    np.random.seed(10)
    dim = 10
    A = np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.T
    A *= np.sign(A)  # get positive matrix
    print(A)
    sigma = 10
    sigmaI = np.eye(dim) * sigma
    phi0 = np.random.randn(dim).T

    w, v = np.linalg.eigh(A)
    residual = np.inf
    phi = phi0.copy()
    phi_old = phi0.copy()
    iter = 0
    iter_max = 1000

    while residual > 1.0E-4 and iter < iter_max:
        # instead of applying (A - sigma I)^{-1} to x_{i} we obtain
        # x_{i+1} by solving (A - sigmaI)x_{i+1} = x_{i} --i.e. Ax = b
        psi = A @ phi
        r = psi.T @ phi
        rvec = psi - r * phi
        phi = psi - sigma * phi
        lam = np.linalg.norm(phi)
        phi /= np.linalg.norm(phi)
        phi_old = phi.copy()
        if iter % 10 == 0:
            print(iter, r, lam, phi.T @ A @ phi / (phi.T @ phi))
        iter += 1
    print("correct sign is lam/x[0,0]", np.sign(lam/phi[0]))

    print(w)

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


    # single walker evolution and sample
    num_samples = 100_000
    possible_walker_indices = list(range(dim))
    sampled_final_output = np.zeros(dim)
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
    print(w)
    print(v)
    # if I want the expectation value of A then I can estimate it from
    # samples of my ground state.  We are sampling output of the power method
    # which is proportional to the ground state where the proportionality  is lambda


def test_gfmc_b_weight():
    dim = 6
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = 2
    for site_idx in range(dim-1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)

    gfmc_inst = DidacticGFMC(hamiltonian, 10_000, 1_000)
    gfmc_inst.build_propagator()
    for ii in range(2**dim):
        assert np.isclose(gfmc_inst.build_weight(ii), sum(abs(gfmc_inst.prop_matrix.getcol(ii).data )))

def test_gfmc_transition_prob():
    """Set up stoquastic matrix propagator and test that we are getting the
    appropriate transition probability"""
    dim = 6
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = -2
    for site_idx in range(dim-1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)

    gfmc_inst = DidacticGFMC(hamiltonian, 10_000, 1_000)
    gfmc_inst.build_propagator()
    A = gfmc_inst.prop_matrix.toarray()
    dim = A.shape[0]
    P = np.zeros((dim, dim))
    Ptest = np.zeros((dim, dim))
    for i, j in product(range(dim), repeat=2):
        if not np.isclose(np.sum(A[:, j]), 0):
            P[i, j] = A[i, j] / np.sum(A[:, j])
        Ptest[i, j] = gfmc_inst.get_transition_prob(i, j)
    assert np.allclose(P, Ptest)


def test_walker_transition():
    dim = 6
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = -2
    for site_idx in range(dim-1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)

    gfmc_inst = DidacticGFMC(hamiltonian, 1000, 1000)
    gfmc_inst.build_propagator()
    A = gfmc_inst.prop_matrix.toarray()
    dim = A.shape[0]
    P = np.zeros((dim, dim))
    weight = np.zeros((dim, dim))
    for i, j in product(range(dim), repeat=2):
        if not np.isclose(np.sum(A[:, j]), 0):
            P[i, j] = A[i, j] / np.sum(A[:, j])
        weight[i, j] = np.sum(A[:, j])

    assert np.allclose(A, weight * P)

    # set up the right fake propagator for this test
    gfmc_inst.get_connected_bitstrings()

    num_trials = 1000
    print()
    for _ in range(num_trials):
        walker_position = np.random.randint(1, dim - 1)
        walker_weight = 1

        walker_i, walker_w = walker_position, walker_weight
        test_transition_probs = [gfmc_inst.get_transition_prob(xx, walker_i) for xx in gfmc_inst.connected_bitstrings[walker_i]]
        assert np.isclose(sum(test_transition_probs), 1)

        # sample new position from column space
        np.random.seed(walker_position)
        new_walker_position = np.random.choice(range(dim), p=P[:, walker_position])
        # update weight and walker
        # walker_weight = weight[new_walker_position, walker_position] * walker_weight
        # walker_position = new_walker_position

        np.random.seed(walker_position)
        test_new_walker_position, test_new_walker_weight = \
            gfmc_inst.transition_walker(walker_position, walker_weight)
        assert np.isclose(test_new_walker_weight, weight[new_walker_position, walker_position] * walker_weight )
        assert test_new_walker_position == new_walker_position

def test_init_walker_weights():
    dim = 6
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = -2
    for site_idx in range(dim-1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)

    gfmc_inst = DidacticGFMC(hamiltonian, 1000, 1000)
    psi_trial = PsiTrial(np.ones(2**dim)/(np.sqrt(2**dim)))
    gfmc_inst.init_walkers(psi_trial=psi_trial)
    assert np.allclose(gfmc_inst.walker_weights, 1)


def test_vmc_energy_estimation():
    """Test the no-propogation estimation"""
    # np.random.seed(10)
    # get Heisnberg lattice
    dim = 4
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = -2
    for site_idx in range(dim - 1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)
    print(hamiltonian)

    gfmc_inst = DidacticGFMC(hamiltonian, 100_000, 10_000)
    A = gfmc_inst.sparse_ham.toarray().real

    w, v = np.linalg.eigh(A)
    I = np.eye(2 ** dim)
    psi_t = abs(v[:, [0]]) + 0.01 * abs(np.random.randn(2 ** dim).reshape((-1, 1)))
    psi_t /= np.linalg.norm(psi_t)
    assert np.isclose(psi_t.conj().T @ psi_t, 1)
    psi_g = psi_t
    sigma = psi_g.conj().T @ A @ psi_g
    G_vals = []
    power_steps = 4
    pwer_energy_steps = []
    for _ in range(power_steps):
        G_vals.append(abs(sigma) * I - A)
        assert np.alltrue(G_vals[-1] >= 0)
        psi_g = (abs(sigma) * I - A) @ psi_g
        # psi_g /= np.linalg.norm(psi_g)
        psi_g_normed = psi_g.copy()
        psi_g_normed /= np.linalg.norm(psi_g_normed)
        eig_est = psi_g_normed.conj().T @ A @ psi_g_normed
        pwer_energy_steps.append(eig_est)
        print(eig_est, eig_est - w[0], np.linalg.norm(psi_g))
        sigmaI = eig_est

    psi_trial = PsiTrial(psi_t)
    psi_trial.generate_Hpsi_t(A)

    gfmc_inst.init_walkers(psi_trial)
    print(gfmc_inst.walker_weights)
    print(gfmc_inst.walker_init_psi_t)
    test_energy = gfmc_inst.batch_energy_estimation(gfmc_inst.walker_idx, gfmc_inst.walker_weights)
    print(test_energy)
    print((psi_t.conj().T @ A @ psi_t)[0, 0].real)
    assert np.isclose(test_energy, (psi_t.conj().T @ A @ psi_t)[0, 0].real,
                      atol=1.0E-3)

    # n_samples = 2_000_000
    # psi_t_dot_H = psi_t.conj().T @ A
    # average_energy_ = []
    # for _ in range(20):
    #     psi_t_samples = np.random.choice(list(range(2**dim)), p=[1/2**dim] * 2**dim, size=n_samples)
    #     numerator = 0
    #     denominator = 0
    #     for pts in psi_t_samples:
    #         numerator += abs(psi_t[pts, 0])**2 * psi_t_dot_H[0, pts] / psi_t[pts, 0]
    #         denominator += abs(psi_t[pts, 0])**2
    #     average_energy_.append(numerator / denominator)
    # print(np.mean(average_energy_), "+-", np.std(average_energy_))
    # assert np.isclose((psi_t.conj().T @ A @ psi_t)[0, 0].real, np.mean(average_energy_), atol=1.0E-3)

def test_path_sum_propagator():
    np.random.seed(10)
    # get Heisnberg lattice
    dim = 4
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = -2
    for site_idx in range(dim - 1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)
    print(hamiltonian)

    gfmc_inst = DidacticGFMC(hamiltonian, 10_000, 10_000)
    A = gfmc_inst.sparse_ham.toarray().real

    w, v = np.linalg.eigh(A)
    I = np.eye(2 ** dim)
    psi_t = abs(v[:, [0]]) + 0.01 * abs(
        np.random.randn(2 ** dim).reshape((-1, 1)))
    psi_t /= np.linalg.norm(psi_t)
    assert np.isclose(psi_t.conj().T @ psi_t, 1)
    psi_g = psi_t
    sigma = psi_g.conj().T @ A @ psi_g
    G_vals = []
    power_steps = 4
    pwer_energy_steps = []
    for _ in range(power_steps):
        G_vals.append(abs(sigma) * I - A)
        assert np.alltrue(G_vals[-1] >= 0)
        psi_g = (abs(sigma) * I - A) @ psi_g
        # psi_g /= np.linalg.norm(psi_g)
        psi_g_normed = psi_g.copy()
        psi_g_normed /= np.linalg.norm(psi_g_normed)
        eig_est = psi_g_normed.conj().T @ A @ psi_g_normed
        pwer_energy_steps.append(eig_est)
        print(eig_est, eig_est - w[0], np.linalg.norm(psi_g))
        sigmaI = eig_est

    def form_stochastic_matrix(G):
        assert np.alltrue(G >= 0)
        P = np.zeros_like(G)
        W = np.zeros_like(G)
        for id, jd in product(range(2 ** dim), repeat=2):
            P[id, jd] = G[id, jd] / np.sum(G[:, jd])
            W[id, jd] = np.sum(G[:, jd])
        return P, W

    # Construct stochastic matrices for each step
    stochastic_mats = []
    weight_mats = []
    for ii in range(len(G_vals)):
        P, W = form_stochastic_matrix(G_vals[ii])
        assert np.allclose(P * W, G_vals[ii])
        assert np.allclose(np.einsum('ij->j', P), 1)
        stochastic_mats.append(P)
        weight_mats.append(W)
    configurations = list(product(range(2 ** dim), repeat=power_steps + 1))


    # numerator = 0
    # denominator = 0
    # psi_t_dot_H = psi_t.conj().T @ A
    # for path in configurations:
    #     g_path_val = 1
    #     for idx in range(power_steps - 1):
    #         g_path_val *= G_vals[idx][path[idx + 1], path[idx]]
    #     g_path_val *= G_vals[-1][path[-1], [path[idx + 1]]]

    #     numerator += (psi_t_dot_H[0, path[-1]] / psi_t[path[-1], 0]) * g_path_val * psi_t[path[-1], 0] * psi_t[path[0], 0]
    #     denominator += g_path_val * psi_t[path[-1], 0] * psi_t[path[0], 0]
    # print(numerator / denominator)
    # print(pwer_energy_steps[1])

    # numerator = 0
    # denominator = 0
    # numerator2 = 0
    # denominator2 = 0
    # total_p_path_val = 0
    # path_probabilties = []
    # psi_t_dot_H = psi_t.conj().T @ A
    # for path in configurations:
    #     g_path_val = psi_t[path[0], 0]
    #     p_path_val = psi_t[path[0], 0]
    #     w_path_val = 1
    #     for idx in range(power_steps - 1):
    #         g_path_val *= G_vals[idx][path[idx + 1], path[idx]]
    #         p_path_val *= stochastic_mats[idx][path[idx + 1], path[idx]]
    #         w_path_val *= weight_mats[idx][path[idx + 1], path[idx]]
    #     p_path_val *= stochastic_mats[-1][path[-1], [path[idx + 1]]] * psi_t[
    #         path[-1], 0]
    #     g_path_val *= G_vals[-1][path[-1], [path[idx + 1]]]
    #     w_path_val *= weight_mats[-1][path[-1], [path[idx + 1]]]
    #     # assert np.isclose(p_path_val * w_path_val, g_path_val * psi_t[path[-1], 0])

    #     numerator += (psi_t_dot_H[0, path[-1]] / psi_t[
    #         path[-1], 0]) * g_path_val * psi_t[path[-1], 0]
    #     denominator += g_path_val * psi_t[path[-1], 0]

    #     numerator2 += (psi_t_dot_H[0, path[-1]] / psi_t[
    #         path[-1], 0]) * p_path_val * w_path_val
    #     denominator2 += p_path_val * w_path_val
    #     total_p_path_val += p_path_val

    # assert np.isclose(numerator / denominator, numerator2 / denominator2)
    # assert np.isclose(pwer_energy_steps[1], numerator / denominator)

    # sum_{n_{0},...,n_{4}=0}^{2**N - 1}
    configurations = list(product(range(2 ** dim), repeat=power_steps + 1))
    final_weights_for_configs = np.zeros(len(configurations))
    for jdx in range(len(configurations)):
        path = configurations[jdx]
        g_path_val = 1 # psi_t[path[0], 0]
        for idx in range(power_steps - 1):
            g_path_val *= G_vals[idx][path[idx + 1], path[idx]]
        g_path_val *= G_vals[-1][path[-1], [path[idx + 1]]] # * psi_t[path[-1], 0]
        final_weights_for_configs[jdx] = g_path_val

    probs_for_configs = final_weights_for_configs / np.sum(final_weights_for_configs)

    trial_result = []
    for _ in range(10):
        n_samples = 1_000_000
        psi_t_dot_H = psi_t.conj().T @ A
        possible_config_idx = list(range(len(configurations)))
        path_idx = np.random.choice(possible_config_idx,
                                    p=probs_for_configs, size=n_samples)
        numerator = 0
        denominator = 0
        energy_samples = []
        for pidx in path_idx:
            path = configurations[pidx]
            numerator +=  (psi_t_dot_H[0, path[-1]] / psi_t[path[-1], 0]) * psi_t[path[0], 0] * psi_t[path[-1], 0]
            denominator += psi_t[path[0], 0] * psi_t[path[-1], 0]
            energy_samples.append(numerator / denominator)
        print(numerator / denominator)
        trial_result.append(numerator / denominator)
    print(np.mean(trial_result))
    print(np.std(trial_result))

    # configurations = list(product(range(2 ** dim), repeat=power_steps + 1))
    # probs_configurations = [1/len(configurations)] * len(configurations)

    # sampled_paths = np.random.choice(range(len(configurations)),
    #                                  p=probs_configurations,
    #                                  size=1_000_000)
    # numerator = 0
    # denominator = 0
    # psi_t_dot_H = psi_t.conj().T @ A
    # for path_index in sampled_paths:
    #     path = configurations[path_index]
    #     g_path_val = 1
    #     for idx in range(power_steps - 1):
    #         g_path_val *= G_vals[idx][path[idx + 1], path[idx]]
    #     g_path_val *= G_vals[-1][path[-1], [path[idx + 1]]]

    #     local_energy = psi_t_dot_H[0, path[-1]] / psi_t[path[-1], 0]
    #     numerator +=  local_energy * g_path_val *  psi_t[path[0], 0] * psi_t[path[-1], 0]
    #     denominator += g_path_val *  psi_t[path[0], 0] * psi_t[path[-1], 0]

    # print(numerator / denominator)

