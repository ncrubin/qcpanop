import fqe

from kUpCCSD import RkUpCCSD
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.openfermion_utils import integrals_to_fqe_restricted
import numpy as np


def fake_fqe_rham(norb):
    fake_h1 = np.zeros((norb, norb))
    fake_h2 = np.zeros((norb,) * 4)
    return RestrictedHamiltonian((fake_h1, fake_h2))


def random_fqe_rham(norb):
    random_h1 = np.random.randn(norb**2).reshape((norb, norb))
    random_h1 = random_h1 + random_h1.T
    random_h2 = np.random.randn(norb**4).reshape((norb,) * 4)
    return RestrictedHamiltonian((random_h1, random_h2))

def test_initialization():
    norb = 4
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    assert isinstance(ansatz, RkUpCCSD)
    assert ansatz.k_layers == 1
    assert ansatz.norb == 4


def test_matricize_variables_k1():
    norb = 5
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)

    x_0_rotation = np.random.random((norb**2)).reshape((norb, norb))
    x_0_rotation = x_0_rotation - x_0_rotation.T
    x_0_nn = np.random.random((norb**2)).reshape((norb, norb))
    x_0_nn = x_0_nn + x_0_nn.T

    row_idx, col_idx = np.triu_indices(norb, k=1)  # row-wise upper triangle
    x_0 = x_0_rotation[row_idx, col_idx].flatten()
    test_rot_mat = np.zeros((norb, norb))
    test_rot_mat[row_idx, col_idx] = x_0
    test_rot_mat = test_rot_mat - test_rot_mat.T
    assert np.allclose(test_rot_mat, x_0_rotation)

    row_idx, col_idx = np.triu_indices(norb, k=0)  # row-wise upper triangle
    x_0 = np.hstack((x_0, x_0_nn[row_idx, col_idx].flatten()))
    [(test_rot_x, test_nn_x)] = ansatz.flattened_params_to_matrices(x_0)

    assert np.allclose(test_rot_x, x_0_rotation)
    assert np.allclose(test_nn_x, x_0_nn)


def test_matricize_variables_k2():
    norb = 6
    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)

    x_0_rotation = np.random.random((norb ** 2)).reshape((norb, norb))
    x_0_rotation = x_0_rotation - x_0_rotation.T
    x_0_nn = np.random.random((norb ** 2)).reshape((norb, norb))
    x_0_nn = x_0_nn + x_0_nn.T

    x_1_rotation = np.random.random((norb ** 2)).reshape((norb, norb))
    x_1_rotation = x_1_rotation - x_1_rotation.T
    x_1_nn = np.random.random((norb ** 2)).reshape((norb, norb))
    x_1_nn = x_1_nn + x_1_nn.T

    row_idx, col_idx = np.triu_indices(norb, k=1)  # row-wise upper triangle
    x_0_rot = x_0_rotation[row_idx, col_idx].flatten()
    x_1_rot = x_1_rotation[row_idx, col_idx].flatten()

    row_idx, col_idx = np.triu_indices(norb, k=0)  # row-wise upper triangle
    x_0_nn_var = x_0_nn[row_idx, col_idx].flatten()
    x_1_nn_var = x_1_nn[row_idx, col_idx].flatten()

    x_0 = np.hstack((x_0_rot, x_0_nn_var, x_1_rot, x_1_nn_var))
    assert x_0.shape[0] == ansatz.k_layers * ansatz.num_params_per_layer
    test_res = ansatz.flattened_params_to_matrices(x_0)
    [(test_rot0_x, test_nn0_x), (test_rot1_x, test_nn1_x)] = test_res

    assert np.allclose(test_rot0_x, x_0_rotation)
    assert np.allclose(test_rot1_x, x_1_rotation)
    assert np.allclose(test_nn0_x, x_0_nn)
    assert np.allclose(test_nn1_x, x_1_nn)

    x_0_test = ansatz.matrices_to_flattened_params(test_res)
    assert np.allclose(x_0_test, x_0)


def test_wavefunction_quadratic_evolution():
    norb = 8
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)

    x0 = ansatz.zero_guess()
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    final_wf = ansatz.wavefunction(x0, init_wf)
    assert np.allclose(final_wf.sector((norb, 0)).coeff,
                       init_wf.sector((norb, 0)).coeff)

    x1 = np.hstack((np.random.randn(ansatz.rotation_params), np.zeros(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[0][0],))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    from fqe.algorithm.low_rank import evolve_fqe_givens
    import copy
    from scipy.linalg import expm
    u = expm(matrix_variables[0][0])
    test_givens_wf = evolve_fqe_givens(copy.deepcopy(init_wf), u)
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_givens_wf.sector((norb, 0)).coeff)

    norb = 8
    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    x0 = ansatz.zero_guess()
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    final_wf = ansatz.wavefunction(x0, init_wf)
    assert np.allclose(final_wf.sector((norb, 0)).coeff,
                       init_wf.sector((norb, 0)).coeff)
    x1 = np.hstack((np.random.randn(ansatz.rotation_params),
                    np.zeros(ansatz.charge_charge_params),
                    np.random.randn(ansatz.rotation_params),
                    np.zeros(ansatz.charge_charge_params)
                     ))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)

    u0 = expm(matrix_variables[0][0])
    u1 = expm(matrix_variables[1][0])
    u = u1 @ u0
    import scipy
    ham_mat = scipy.linalg.logm(u)
    fqe_quad_op = RestrictedHamiltonian((1j * ham_mat,))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    test_givens_wf = evolve_fqe_givens(copy.deepcopy(init_wf), u0)
    test_givens_wf = evolve_fqe_givens(test_givens_wf, u1)
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_givens_wf.sector((norb, 0)).coeff)


def test_wavefunction_coulomb_evolution():
    norb = 8
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)

    x0 = ansatz.zero_guess()
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    final_wf = ansatz.wavefunction(x0, init_wf)
    assert np.allclose(final_wf.sector((norb, 0)).coeff,
                       init_wf.sector((norb, 0)).coeff)

    x1 = np.hstack((np.zeros(ansatz.rotation_params), np.random.randn(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(init_wf, matrix_variables[0][1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    norb = 8
    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    x0 = ansatz.zero_guess()
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    final_wf = ansatz.wavefunction(x0, init_wf)
    assert np.allclose(final_wf.sector((norb, 0)).coeff,
                       init_wf.sector((norb, 0)).coeff)
    x1 = np.hstack((np.zeros(ansatz.rotation_params),
                    np.random.randn(ansatz.charge_charge_params),
                    np.zeros(ansatz.rotation_params),
                    np.random.randn(ansatz.charge_charge_params)
                     ))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)

    true_final_wf = evolve_fqe_charge_charge_alpha_beta(init_wf,
                                                        matrix_variables[0][1])
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(true_final_wf,
                                                        matrix_variables[1][1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)


def test_wavefunction_evolution():
    norb = 8
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)

    x0 = ansatz.zero_guess()
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    final_wf = ansatz.wavefunction(x0, init_wf)
    assert np.allclose(final_wf.sector((norb, 0)).coeff,
                       init_wf.sector((norb, 0)).coeff)

    x1 = np.hstack((np.random.randn(ansatz.rotation_params),
                    np.random.randn(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[0][0],))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(true_final_wf,
                                                        matrix_variables[0][
                                                            1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)


def test_one_body_gradient():
    np.random.seed(25)
    norb = 6
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')

    rfqe_ham = random_fqe_rham(norb)

    x1 = np.hstack((np.random.randn(ansatz.rotation_params), np.zeros(ansatz.charge_charge_params)))
    # x1 = np.hstack((np.zeros(ansatz.rotation_params), np.zeros(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[0][0],))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    eps = 1.0E-3
    abs_diffs = []
    for eps in [1.E-4]: # np.logspace(-2, -4, 15):
        for xx in range(ansatz.rotation_params):
            grad_x = np.hstack((np.zeros(ansatz.rotation_params),
                                np.zeros(ansatz.charge_charge_params)))
            grad_x[xx] = 1
            test_grad_wf = ansatz.gradient(grad_x, x1, init_wf)

            # print()
            plus_eps = x1.copy()
            plus_eps[xx] += eps
            plus_wf = ansatz.wavefunction(plus_eps, init_wf)
            minus_eps = x1.copy()
            minus_eps[xx] -= eps
            minus_wf = ansatz.wavefunction(minus_eps, init_wf)
            fd_grad_wf = plus_wf - minus_wf
            fd_grad_wf.scale(1/ (2 * eps))

            diff = test_grad_wf - fd_grad_wf
            abs_diffs.append(np.linalg.norm(diff.sector((norb, 0)).coeff))

    assert np.allclose(abs_diffs, 0, atol=1.0E-6)
    # import matplotlib.pyplot as plt
    # plt.semilogy(range(len(abs_diffs)), abs_diffs, 'C0o-')
    # plt.show()


def test_two_body_gradient():
    np.random.seed(25)
    norb = 6
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    # init_wf.set_wfn(strategy='hartree-fock')
    init_wf.set_wfn(strategy='random')

    rfqe_ham = random_fqe_rham(norb)

    x1 = np.hstack((np.zeros(ansatz.rotation_params), np.random.randn(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(init_wf, matrix_variables[0][1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    # print("Charge Charge")
    abs_diffs = []
    for eps in [1.0E-4]: # np.logspace(-2, -4, 15):
        for xx in range(ansatz.charge_charge_params):
            grad_x = np.hstack((np.zeros(ansatz.rotation_params),
                                np.zeros(ansatz.charge_charge_params)))
            grad_x[ansatz.rotation_params + xx] = 1
            test_grad_wf = ansatz.gradient(grad_x, x1, init_wf)
            # print("Test Gradient")
            # test_grad_wf.print_wfn()

            # print()
            plus_eps = x1.copy()
            plus_eps[ansatz.rotation_params + xx] += eps
            plus_wf = ansatz.wavefunction(plus_eps, init_wf)
            # plus_wf.print_wfn()
            minus_eps = x1.copy()
            minus_eps[ansatz.rotation_params + xx] -= eps
            minus_wf = ansatz.wavefunction(minus_eps, init_wf)
            # plus_wf.print_wfn()
            fd_grad_wf = plus_wf - minus_wf
            fd_grad_wf.scale(1 / (2 * eps))
            # print("Finite difference grad")
            # fd_grad_wf.print_wfn(threshold=1.0E-12)

            diff = test_grad_wf - fd_grad_wf
            abs_diffs.append(np.linalg.norm(diff.sector((norb, 0)).coeff))
            # print("absolute diff ", abs_diffs[-1])

    assert np.allclose(abs_diffs, 0, atol=1.0E-6)
    # import matplotlib.pyplot as plt
    # plt.semilogy(range(len(abs_diffs)), abs_diffs, 'C0o-')
    # plt.show()


def test_full_gradient():
    np.random.seed(25)
    norb = 6
    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    # init_wf.set_wfn(strategy='hartree-fock')
    init_wf.set_wfn(strategy='random')

    rfqe_ham = random_fqe_rham(norb)

    x1 = np.hstack((np.random.randn(ansatz.rotation_params), np.random.randn(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[0][0],))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(true_final_wf, matrix_variables[0][1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    # print("Charge Charge")
    abs_diffs = []
    for eps in [1.0E-4]: # np.logspace(-2, -4, 15):
        for xx in range(ansatz.charge_charge_params + ansatz.rotation_params):
            grad_x = np.hstack((np.zeros(ansatz.rotation_params),
                                np.zeros(ansatz.charge_charge_params)))
            grad_x[xx] = 1
            test_grad_wf = ansatz.gradient(grad_x, x1, init_wf)
            # print("Test Gradient")
            # test_grad_wf.print_wfn()

            # print()
            plus_eps = x1.copy()
            plus_eps[xx] += eps
            plus_wf = ansatz.wavefunction(plus_eps, init_wf)
            # plus_wf.print_wfn()
            minus_eps = x1.copy()
            minus_eps[xx] -= eps
            minus_wf = ansatz.wavefunction(minus_eps, init_wf)
            # plus_wf.print_wfn()
            fd_grad_wf = plus_wf - minus_wf
            fd_grad_wf.scale(1 / (2 * eps))
            # print("Finite difference grad")
            # fd_grad_wf.print_wfn(threshold=1.0E-12)

            diff = test_grad_wf - fd_grad_wf
            abs_diffs.append(np.linalg.norm(diff.sector((norb, 0)).coeff))
            # print("absolute diff ", abs_diffs[-1])

    assert np.allclose(abs_diffs, 0, atol=1.0E-6)
    # import matplotlib.pyplot as plt
    # plt.semilogy(range(len(abs_diffs)), abs_diffs, 'C0o-')
    # plt.show()


def test_two_layer_full_gradient():
    np.random.seed(25)
    norb = 6
    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    init_wf = fqe.Wavefunction([[norb, 0, norb]])
    # init_wf.set_wfn(strategy='hartree-fock')
    init_wf.set_wfn(strategy='random')

    rfqe_ham = random_fqe_rham(norb)

    x1 = np.hstack((np.random.randn(ansatz.rotation_params), np.random.randn(ansatz.charge_charge_params),
                    np.random.randn(ansatz.rotation_params), np.random.randn(ansatz.charge_charge_params)))
    test_final_wf = ansatz.wavefunction(x1, init_wf)
    matrix_variables = ansatz.flattened_params_to_matrices(x1)
    from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[0][0],))
    true_final_wf = init_wf.time_evolve(1.0, fqe_quad_op)
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(true_final_wf, matrix_variables[0][1])
    fqe_quad_op = RestrictedHamiltonian((1j * matrix_variables[1][0],))
    true_final_wf = true_final_wf.time_evolve(1.0, fqe_quad_op)
    true_final_wf = evolve_fqe_charge_charge_alpha_beta(true_final_wf,
                                                        matrix_variables[1][1])
    assert np.allclose(true_final_wf.sector((norb, 0)).coeff,
                       test_final_wf.sector((norb, 0)).coeff)

    # print("Charge Charge")
    abs_diffs = []
    for eps in [1.0E-4]: # np.logspace(-2, -4, 15):
        for xx in range(2 * (ansatz.charge_charge_params + ansatz.rotation_params)):
            grad_x = np.hstack((np.zeros(ansatz.rotation_params),
                                np.zeros(ansatz.charge_charge_params),
                                np.zeros(ansatz.rotation_params),
                                np.zeros(ansatz.charge_charge_params)
                                ))
            grad_x[xx] = 1
            test_grad_wf = ansatz.gradient(grad_x, x1, init_wf)
            # print("Test Gradient")
            # test_grad_wf.print_wfn()

            # print()
            plus_eps = x1.copy()
            plus_eps[xx] += eps
            plus_wf = ansatz.wavefunction(plus_eps, init_wf)
            # plus_wf.print_wfn()
            minus_eps = x1.copy()
            minus_eps[xx] -= eps
            minus_wf = ansatz.wavefunction(minus_eps, init_wf)
            # plus_wf.print_wfn()
            fd_grad_wf = plus_wf - minus_wf
            fd_grad_wf.scale(1 / (2 * eps))
            # print("Finite difference grad")
            # fd_grad_wf.print_wfn(threshold=1.0E-12)

            diff = test_grad_wf - fd_grad_wf
            abs_diffs.append(np.linalg.norm(diff.sector((norb, 0)).coeff))
            # print("absolute diff ", abs_diffs[-1])

    assert np.allclose(abs_diffs, 0, atol=1.0E-6)
    #  import matplotlib.pyplot as plt
    #  plt.semilogy(range(len(abs_diffs)), abs_diffs, 'C0o-')
    #  plt.show()


def test_cost_func_grad():
    np.random.seed(5)
    from openfermion.testing.lih_integration_test import LiHIntegrationTest
    lih_mol = LiHIntegrationTest()
    lih_mol.setUp()
    molecule = lih_mol.molecule
    oei, tei = molecule.get_integrals()
    norb = oei.shape[0]
    nelec = molecule.n_electrons
    sz = 0

    fqe_ham = integrals_to_fqe_restricted(oei, tei)
    fqe_ham._e_0 = molecule.nuclear_repulsion
    init_wf = fqe.Wavefunction([[nelec, sz, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    assert np.isclose(init_wf.expectationValue(fqe_ham).real, molecule.hf_energy)

    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    obj_val, grad_val = ansatz.gradient_obj(random_params, init_wf, fqe_ham)
    true_obj_val = ansatz.wavefunction(random_params, init_wf).expectationValue(fqe_ham).real

    assert np.isclose(obj_val, true_obj_val)
    eps = 1.0E-4
    for xx in range(ansatz.k_layers * ansatz.num_params_per_layer):
        plus_eps = random_params.copy()
        plus_eps[xx] += eps
        plus_wf = ansatz.wavefunction(plus_eps, init_wf)
        plus_val = plus_wf.expectationValue(fqe_ham).real

        minus_eps = random_params.copy()
        minus_eps[xx] -= eps
        minus_wf = ansatz.wavefunction(minus_eps, init_wf)
        minus_val = minus_wf.expectationValue(fqe_ham).real

        fd_grad_val = (plus_val - minus_val) / (2 * eps)

        assert np.isclose(grad_val[xx], fd_grad_val, atol=1.0E-6)

    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    obj_val, grad_val = ansatz.gradient_obj(random_params, init_wf, fqe_ham)
    true_obj_val = ansatz.wavefunction(random_params, init_wf).expectationValue(fqe_ham).real

    assert np.isclose(obj_val, true_obj_val)
    eps = 1.0E-4
    for xx in range(ansatz.k_layers * ansatz.num_params_per_layer):
        plus_eps = random_params.copy()
        plus_eps[xx] += eps
        plus_wf = ansatz.wavefunction(plus_eps, init_wf)
        plus_val = plus_wf.expectationValue(fqe_ham).real

        minus_eps = random_params.copy()
        minus_eps[xx] -= eps
        minus_wf = ansatz.wavefunction(minus_eps, init_wf)
        minus_val = minus_wf.expectationValue(fqe_ham).real

        fd_grad_val = (plus_val - minus_val) / (2 * eps)

        assert np.isclose(grad_val[xx], fd_grad_val, atol=1.0E-6)


def test_cost_func_grad_backprop():
    np.random.seed(5)
    from openfermion.testing.lih_integration_test import LiHIntegrationTest
    lih_mol = LiHIntegrationTest()
    lih_mol.setUp()
    molecule = lih_mol.molecule
    oei, tei = molecule.get_integrals()
    norb = oei.shape[0]
    nelec = molecule.n_electrons
    sz = 0

    fqe_ham = integrals_to_fqe_restricted(oei, tei)
    fqe_ham._e_0 = molecule.nuclear_repulsion
    init_wf = fqe.Wavefunction([[nelec, sz, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    assert np.isclose(init_wf.expectationValue(fqe_ham).real, molecule.hf_energy)

    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    obj_val, grad_val = ansatz.gradient_backprop(random_params, init_wf, fqe_ham)
    true_obj_val = ansatz.wavefunction(random_params, init_wf).expectationValue(fqe_ham).real

    assert np.isclose(obj_val, true_obj_val)
    eps = 1.0E-4
    for xx in range(ansatz.k_layers * ansatz.num_params_per_layer):
        plus_eps = random_params.copy()
        plus_eps[xx] += eps
        plus_wf = ansatz.wavefunction(plus_eps, init_wf)
        plus_val = plus_wf.expectationValue(fqe_ham).real

        minus_eps = random_params.copy()
        minus_eps[xx] -= eps
        minus_wf = ansatz.wavefunction(minus_eps, init_wf)
        minus_val = minus_wf.expectationValue(fqe_ham).real

        fd_grad_val = (plus_val - minus_val) / (2 * eps)

        assert np.isclose(grad_val[xx], fd_grad_val, atol=1.0E-6)

    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    obj_val, grad_val = ansatz.gradient_backprop(random_params, init_wf, fqe_ham)
    true_obj_val = ansatz.wavefunction(random_params, init_wf).expectationValue(fqe_ham).real

    assert np.isclose(obj_val, true_obj_val)
    eps = 1.0E-4
    for xx in range(ansatz.k_layers * ansatz.num_params_per_layer):
        plus_eps = random_params.copy()
        plus_eps[xx] += eps
        plus_wf = ansatz.wavefunction(plus_eps, init_wf)
        plus_val = plus_wf.expectationValue(fqe_ham).real

        minus_eps = random_params.copy()
        minus_eps[xx] -= eps
        minus_wf = ansatz.wavefunction(minus_eps, init_wf)
        minus_val = minus_wf.expectationValue(fqe_ham).real

        fd_grad_val = (plus_val - minus_val) / (2 * eps)

        assert np.isclose(grad_val[xx], fd_grad_val, atol=1.0E-6)


def test_timing_backprop_vs_normalgrad():
    np.random.seed(5)
    from openfermion.testing.lih_integration_test import LiHIntegrationTest
    import time
    lih_mol = LiHIntegrationTest()
    lih_mol.setUp()
    molecule = lih_mol.molecule
    oei, tei = molecule.get_integrals()
    norb = oei.shape[0]
    nelec = molecule.n_electrons
    sz = 0

    fqe_ham = integrals_to_fqe_restricted(oei, tei)
    fqe_ham._e_0 = molecule.nuclear_repulsion
    init_wf = fqe.Wavefunction([[nelec, sz, norb]])
    init_wf.set_wfn(strategy='hartree-fock')
    assert np.isclose(init_wf.expectationValue(fqe_ham).real, molecule.hf_energy)

    k_layers = 1
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    start_time_bp = time.time()
    obj_val, grad_val = ansatz.gradient_backprop(random_params, init_wf, fqe_ham)
    end_time_bp = time.time()
    start_time_ng = time.time()
    obj_val, grad_val = ansatz.gradient_obj(random_params, init_wf, fqe_ham)
    end_time_ng = time.time()
    print("k = 1 relative gradient timings")
    print("Backprop timing {}".format(end_time_bp - start_time_bp))
    print("normal grad timing {}".format(end_time_ng - start_time_ng))

    k_layers = 2
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    print("k = 2 relative gradient timings")
    start_time_bp = time.time()
    obj_val, grad_val = ansatz.gradient_backprop(random_params, init_wf, fqe_ham)
    end_time_bp = time.time()
    start_time_ng = time.time()
    obj_val, grad_val = ansatz.gradient_obj(random_params, init_wf, fqe_ham)
    end_time_ng = time.time()
    print("Backprop timing {}".format(end_time_bp - start_time_bp))
    print("normal grad timing {}".format(end_time_ng - start_time_ng))

    k_layers = 3
    ansatz = RkUpCCSD(norb, k_layers)
    random_params = np.random.randn(len(ansatz.zero_guess()))

    print("k = 3 relative gradient timings")
    start_time_bp = time.time()
    obj_val, grad_val = ansatz.gradient_backprop(random_params, init_wf, fqe_ham)
    end_time_bp = time.time()
    start_time_ng = time.time()
    obj_val, grad_val = ansatz.gradient_obj(random_params, init_wf, fqe_ham)
    end_time_ng = time.time()
    print("Backprop timing {}".format(end_time_bp - start_time_bp))
    print("normal grad timing {}".format(end_time_ng - start_time_ng))


if __name__ == "__main__":
    test_initialization()
    test_matricize_variables_k1()
    test_matricize_variables_k2()
    test_wavefunction_quadratic_evolution()
    test_wavefunction_coulomb_evolution()
    test_wavefunction_evolution()
    test_one_body_gradient()
    test_two_body_gradient()
    test_full_gradient()
    test_two_layer_full_gradient()
    test_cost_func_grad()
    test_cost_func_grad_backprop()
    test_timing_backprop_vs_normalgrad()
    print("TESTS PASSED")