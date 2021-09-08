import copy
import numpy as np
from itertools import product

import fqe
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.algorithm.low_rank import evolve_fqe_charge_charge_alpha_beta

import openfermion as of


def get_matrix_of_eigs(w):
    """
    Transform the eigenvalues for getting the gradient

    .. math:
        f(w) \rightarrow \frac{e^{i (\lambda_{i} - \lambda_{j})}{i (\lambda_{i} - \lambda_{j})}

    :param w:
    :return:
    """
    transform_eigs = np.zeros((w.shape[0], w.shape[0]),
                               dtype=np.complex128)
    for i, j in product(range(w.shape[0]), repeat=2):
        if np.isclose(abs(w[i] - w[j]), 0):
            transform_eigs[i, j] = 1
        else:
            transform_eigs[i, j] = (np.exp(1j * (w[i] - w[j])) - 1) / (
                        1j * (w[i] - w[j]))
    return transform_eigs


def quadratic_gradient_prematrix(generator_mat, a, b):
    w_full, v_full = np.linalg.eigh(
        -1j * generator_mat)  # so that kappa = i U lambda U^
    eigs_scaled_full = get_matrix_of_eigs(w_full)

    Y_full = np.zeros_like(generator_mat, dtype=np.complex128)
    if a == b:
        Y_full[a, b] = 0
    else:
        Y_full[a, b] = 1.0
        Y_full[b, a] = -1.0

    Y_kl_full = v_full.conj().T @ Y_full @ v_full
    # now rotate Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
    # into the original basis
    pre_matrix_full = v_full @ (
            eigs_scaled_full * Y_kl_full) @ v_full.conj().T

    return pre_matrix_full


class RkUpCCSD:

    def __init__(self, norb: int, k_layers: int):
        self.norb = norb
        self.k_layers = k_layers

        #### Do not set unless expert
        self.rotation_params = self.norb * (self.norb - 1) // 2
        self.charge_charge_params = self.norb * (self.norb + 1) // 2
        self.num_params_per_layer = self.rotation_params + \
                                    self.charge_charge_params
        self.k_1_triu = np.triu_indices(self.norb, k=1)
        self.k_0_triu = np.triu_indices(self.norb, k=0)

    def zero_guess(self):
        guess_matrices = []
        zero_mat = np.zeros((self.norb, self.norb))
        for kk in range(self.k_layers):
            guess_matrices.append((zero_mat.copy(), zero_mat.copy()))
        return self.matrices_to_flattened_params(guess_matrices)


    def matrices_to_flattened_params(self, list_variables):
        """
        Turn list of tuple of matrices into parameter vector

        For each tuple the first matrix is the rotation matrix and
        should be real antisymmetric. The second matrix is the charge-charge and
        should be symmetric real.
        """
        x_0_rot = list_variables[0][0][self.k_1_triu].flatten()
        x_0_nn = list_variables[0][1][self.k_0_triu].flatten()
        x_0 = np.hstack((x_0_rot, x_0_nn))

        if len(list_variables) == 1:
            return x_0

        for (rot_mat, nn_mat) in list_variables[1:]:
            x_kk_rot = rot_mat[self.k_1_triu].flatten()
            x_kk_nn = nn_mat[self.k_0_triu].flatten()
            x_0 = np.hstack((x_0, x_kk_rot, x_kk_nn))
        return x_0

    def flattened_params_to_matrices(self, x):
        """Convert parameter vector to matrices

        vector stores variables in row-wise order.

        [x_{0, 0} = M_{0, 1}, x_{0, 1}, M_{0, 2},...
        :param np.ndarray x: Parameter vector
        :returns: self.k_layers tuple of matrices matrices
        """
        layer_nvars = self.num_params_per_layer
        matricized_variables = []
        for kk in range(self.k_layers):
            # set empty matrices
            rotation_var_mat = np.zeros((self.norb, self.norb))
            nn_var_mat = np.zeros((self.norb, self.norb))

            # grab variables
            layer_variables = x[layer_nvars * kk:layer_nvars * (kk + 1)]
            rotation_vars = layer_variables[:self.rotation_params]
            charge_charge_vars = layer_variables[self.rotation_params:]

            # fill matrices
            rotation_var_mat[self.k_1_triu] = rotation_vars
            nn_var_mat[self.k_0_triu] = charge_charge_vars

            # distribute variables according to what makes hermitian ops
            rotation_var_mat = rotation_var_mat - rotation_var_mat.T
            diagonals = np.diagonal(nn_var_mat)
            nn_var_mat = nn_var_mat + nn_var_mat.T
            np.fill_diagonal(nn_var_mat, diagonals)

            # populate returned variable
            matricized_variables.append((rotation_var_mat, nn_var_mat))

        return matricized_variables

    def wavefunction(self, parameters, init_wf: fqe.Wavefunction) -> fqe.Wavefunction:
        """
        Return an FQE wavefunction
        """
        matricized_variables = self.flattened_params_to_matrices(parameters)
        current_wf = copy.deepcopy(init_wf)
        for (rot_mat, nn_mat) in matricized_variables:
            fqe_quad_ham = RestrictedHamiltonian((1j * rot_mat,))
            current_wf.time_evolve(1, fqe_quad_ham, inplace=True)
            current_wf = evolve_fqe_charge_charge_alpha_beta(current_wf, nn_mat)
        return current_wf

    def gradient(self, grad_position, parameters, init_wf: fqe.Wavefunction):
        """
        Calculate the gradient in the slow (quadratic scaling) fashion.

        This should be used for testing purposes against the backprop version.
        """
        # check we have only one gradient to calculate
        assert len(np.where(np.isclose(grad_position, 1))[0])  == 1
        matricized_grad_position = self.flattened_params_to_matrices(grad_position)
        matricized_variables = self.flattened_params_to_matrices(parameters)
        current_wf = copy.deepcopy(init_wf)
        for kidx, (rot_mat, nn_mat) in enumerate(matricized_variables):
            # evolve rot_mat
            fqe_quad_ham = RestrictedHamiltonian((1j * rot_mat,))
            current_wf.time_evolve(1, fqe_quad_ham, inplace=True)
            # check if matriced-grad-position has one in this k
            if len(np.where(np.isclose(matricized_grad_position[kidx][0], 1))[0]) != 0:
                grad_row_idx, grad_col_idx = np.where(np.isclose(matricized_grad_position[kidx][0], 1))
                grad_row_idx, grad_col_idx = grad_row_idx[0], grad_col_idx[0]
                pre_matrix = quadratic_gradient_prematrix(rot_mat, grad_row_idx, grad_col_idx)
                assert of.is_hermitian(1j * pre_matrix)
                fqe_quad_ham_pre = RestrictedHamiltonian((pre_matrix,))
                current_wf = current_wf.apply(fqe_quad_ham_pre)
                # current_wf.scale(-1j)

            current_wf = evolve_fqe_charge_charge_alpha_beta(current_wf, nn_mat)

            if len(np.where(np.isclose(matricized_grad_position[kidx][1], 1))[0]) != 0:
                grad_row_idx, grad_col_idx = np.where(np.isclose(matricized_grad_position[kidx][1], 1))
                p, q = int(grad_row_idx[0]), int(grad_col_idx[0])
                if p != q:
                    fop = of.FermionOperator(((2 * p, 1), (2 * p, 0), (2 * q + 1, 1), (2 * q + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    current_wf_1 = current_wf.apply(fqe_op)

                    fop = of.FermionOperator(((2 * q, 1), (2 * q, 0), (2 * p + 1, 1), (2 * p + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    current_wf_2 = current_wf.apply(fqe_op)
                    current_wf = current_wf_1 + current_wf_2
                else:

                    fop = of.FermionOperator(((2 * p, 1), (2 * p, 0), (2 * q + 1, 1), (2 * q + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    current_wf = current_wf.apply(fqe_op)
                current_wf.scale(-1j)

        return current_wf

    def gradient_obj(self, parameters, init_wf, objective_operator):
        """
        Compute gradient and objective slow way
        """
        phi_state = self.wavefunction(parameters, init_wf)
        cost_val = phi_state.expectationValue(objective_operator).real
        grad_vec = np.zeros_like(parameters)
        for xx in range(len(grad_vec)):
            grad_x = self.zero_guess()
            grad_x[xx] = 1
            grad_wf = self.gradient(grad_x, parameters, init_wf)
            lambda_wf = grad_wf.apply(objective_operator)
            grad_val = fqe.vdot(phi_state, lambda_wf)
            grad_vec[xx] = 2 * grad_val.real
        return cost_val, grad_vec

    def gradient_backprop(self, parameters, init_wf: fqe.Wavefunction, objective_operator):
        """
        Compute the gradient via backprop given an objective operator

        This is a WIP
        """
        phi_state = self.wavefunction(parameters, init_wf)
        cost_val = phi_state.expectationValue(objective_operator).real
        lambda_state = phi_state.apply(objective_operator)
        matricized_variables = self.flattened_params_to_matrices(parameters)
        grad_position = self.k_layers * self.num_params_per_layer - 1
        grad_vec = np.zeros_like(parameters)

        # NOTE: enumerate is not sliceable so kidx is same as before. Need to
        # adjust to kmax - kidx
        for kidx, (rot_mat, nn_mat) in enumerate(matricized_variables[::-1]):
            # negative is dagger
            # |phi> = U_{k}^{\dagger}|phi>
            # NOTE: Now that we lam - U_{k}^{\dagge}U_{k+1}^{\dagger}...O...U_{k+1}U_{k}...
            # we are interested in U_{k-1}(-iP_{k})U_{K}^{\dagger}U_{k+1}^{\dagger}...O...U_{k+1}U_{k}U_{k-1}...
            # so for each mu copy the current phi and act on it by -iP_{k}
            # then compute overlaps

            phi_state = evolve_fqe_charge_charge_alpha_beta(phi_state, -nn_mat)
            lambda_state = evolve_fqe_charge_charge_alpha_beta(lambda_state, -nn_mat)

            # now make a norb * (norb + 1) // 2 states
            # where each states gets iP_{nn_idx}|phi>
            for _ in range(self.charge_charge_params):
                mu_state = copy.deepcopy(phi_state)
                zero_params_for_grad = self.zero_guess()
                zero_params_for_grad[grad_position] = 1
                matricized_grad_position = self.flattened_params_to_matrices(
                    zero_params_for_grad)
                grad_row_idx, grad_col_idx = np.where(np.isclose(matricized_grad_position[self.k_layers - kidx - 1][1], 1))
                p, q = int(grad_row_idx[0]), int(grad_col_idx[0])
                if p != q:
                    fop = of.FermionOperator(
                        ((2 * p, 1), (2 * p, 0), (2 * q + 1, 1), (2 * q + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    mu_state_1 = mu_state.apply(fqe_op)

                    fop = of.FermionOperator(
                        ((2 * q, 1), (2 * q, 0), (2 * p + 1, 1), (2 * p + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    mu_state_2 = mu_state.apply(fqe_op)
                    mu_state = mu_state_1 + mu_state_2
                else:
                    fop = of.FermionOperator(
                        ((2 * p, 1), (2 * p, 0), (2 * q + 1, 1), (2 * q + 1, 0)))
                    fqe_op = fqe.build_hamiltonian(fop, norb=self.norb,
                                                   conserve_number=True)
                    mu_state = mu_state.apply(fqe_op)
                mu_state.scale(-1j)

                grad_vec[grad_position] = 2 * fqe.vdot(lambda_state, mu_state).real
                grad_position -= 1

            fqe_quad_ham = RestrictedHamiltonian((1j * -rot_mat,))
            phi_state.time_evolve(1, fqe_quad_ham, inplace=True)
            lambda_state.time_evolve(1, fqe_quad_ham, inplace=True)

            for _ in range(self.rotation_params):
                mu_state = copy.deepcopy(phi_state)
                zero_params_for_grad = self.zero_guess()
                zero_params_for_grad[grad_position] = 1
                matricized_grad_position = self.flattened_params_to_matrices(
                    zero_params_for_grad)
                grad_row_idx, grad_col_idx = np.where(np.isclose(matricized_grad_position[self.k_layers - kidx - 1][0], 1))
                grad_row_idx, grad_col_idx = grad_row_idx[0], grad_col_idx[0]
                # minus sign because of order of kappa matrix operator
                pre_matrix = quadratic_gradient_prematrix(-rot_mat, grad_row_idx, grad_col_idx)
                assert of.is_hermitian(1j * pre_matrix)
                fqe_quad_ham_pre = RestrictedHamiltonian((pre_matrix,))
                mu_state = mu_state.apply(fqe_quad_ham_pre)
                grad_vec[grad_position] = 2 * fqe.vdot(lambda_state, mu_state).real
                grad_position -= 1

        return cost_val, grad_vec