"""
Gradient based orbital optimizations from the perspective of

exp(kappa)
kappa = k_{p<q}(p^q - q^p)

1) Gradient only where we provide just the gradient with respect to the
   orbital rotation parameters.  This saves us from having to re-rotate
   our basis at every step of the algorithm.  The cost is now in computing
   the gradient with the 1-RDM and 2-RDM provided.  Graident cost is now


2) A Newton type method where we calculate the gradient and hessian
   around kappa = 0.
"""
from typing import List, Tuple, Dict, Optional, Union

import os

import numpy as np
import scipy as sp
from scipy.linalg import expm

from itertools import product

from functools import reduce

import copy

import fqe
from fqe.algorithm.brillouin_calculator import get_fermion_op
from fqe.algorithm.low_rank import evolve_fqe_givens
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

import openfermion as of
from openfermion.config import DATA_DIRECTORY
from openfermion.chem.molecular_data import spinorb_from_spatial

from openfermionpyscf import run_pyscf

import matplotlib.pyplot as plt


def load_hydrogen_fluoride_molecule(bd):
    geometry = [('H', (0., 0., 0.)), ('Li', (0., 0., bd))]
    basis = 'sto-3g'
    multiplicity = 1
    molecule = of.MolecularData(geometry, basis, multiplicity)
    molecule = run_pyscf(molecule, run_scf=True)# , run_fci=True, run_cisd=True,
                         # run_mp2=True)
    molecule.filename = os.path.join(os.getcwd(), molecule.name)
    # # molecule.save()
    # molecule.load()
    return molecule



def rhf_params_to_matrix(parameters: np.ndarray,
                         num_qubits: int,
                         occ: Optional[Union[None, List[int]]] = None,
                         virt: Optional[Union[None, List[int]]] = None):
    """Assemble variational parameters into a matrix.

    For restricted Hartree-Fock we have nocc * nvirt parameters. These are
    provided as a list that is ordered by (virtuals) \times (occupied) where
    occupied is a set of indices corresponding to the occupied orbitals w.r.t
    the Lowdin basis and virtuals is a set of indices of the virtual orbitals
    w.r.t the Lowdin basis.  For example, for H4 we have 2 orbitals occupied and
    2 virtuals:

    occupied = [0, 1]  virtuals = [2, 3]

    parameters = [(v_{0}, o_{0}), (v_{0}, o_{1}), (v_{1}, o_{0}),
                  (v_{1}, o_{1})]
               = [(2, 0), (2, 1), (3, 0), (3, 1)]

    You can think of the tuples of elements of the upper right triangle of the
    antihermitian matrix that specifies the c_{b, i} coefficients.

    coefficient matrix
    [[ c_{0, 0}, -c_{1, 0}, -c_{2, 0}, -c_{3, 0}],
     [ c_{1, 0},  c_{1, 1}, -c_{2, 1}, -c_{3, 1}],
     [ c_{2, 0},  c_{2, 1},  c_{2, 2}, -c_{3, 2}],
     [ c_{3, 0},  c_{3, 1},  c_{3, 2},  c_{3, 3}]]

    Since we are working with only non-redundant operators we know c_{i, i} = 0
    and any c_{i, j} where i and j are both in occupied or both in virtual = 0.
    """
    if occ is None:
        occ = range(num_qubits // 2)
    if virt is None:
        virt = range(num_qubits // 2, num_qubits)

    # check that parameters are a real array
    if not np.allclose(parameters.imag, 0):
        raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa


def get_matrix_of_eigs(w: np.ndarray) -> np.ndarray:
    r"""Transform the eigenvalues for getting the gradient.

    .. math:
        f(w) \rightarrow
        \frac{e^{i (\lambda_{i} - \lambda_{j})}}{i (\lambda_{i} - \lambda_{j})}

    Args:
        w: eigenvalues of C-matrix

    Returns:
        New array of transformed eigenvalues
    """
    transform_eigs = np.zeros((w.shape[0], w.shape[0]), dtype=np.complex128)
    for i, j in product(range(w.shape[0]), repeat=2):
        if np.isclose(abs(w[i] - w[j]), 0):
            transform_eigs[i, j] = 1
        else:
            transform_eigs[i, j] = (np.exp(1j *
                                           (w[i] - w[j])) - 1) / (1j *
                                                                  (w[i] - w[j]))
    return transform_eigs


class RestrictedHartreeFockObjective():
    """Objective function for Restricted Hartree-Fock.

    The object transforms a variety of input types into the appropriate output.
    It does this by analyzing the type and size of the input based on its
    knowledge of each type.
    """

    def __init__(self, hamiltonian: of.InteractionOperator, num_electrons: int):
        self.hamiltonian = hamiltonian
        self.fermion_hamiltonian = of.get_fermion_operator(self.hamiltonian)
        self.num_qubits = hamiltonian.one_body_tensor.shape[0]
        self.num_orbitals = self.num_qubits // 2
        self.num_electrons = num_electrons
        self.nocc = self.num_electrons // 2
        self.nvirt = self.num_orbitals - self.nocc
        self.occ = list(range(self.nocc))
        self.virt = list(range(self.nocc, self.nocc + self.nvirt))


    def rdms_from_opdm_aa(self, opdm_aa) -> of.InteractionRDM:
        """Generate the RDM from just the alpha-alpha block.

        Due to symmetry, the beta-beta block is the same, and the other
        blocks are zero.

        Args:
            opdm_aa: The alpha-alpha block of the RDM
        """
        opdm = np.zeros((self.num_qubits, self.num_qubits), dtype=complex)
        opdm[::2, ::2] = opdm_aa
        opdm[1::2, 1::2] = opdm_aa
        tpdm = of.wedge(opdm, opdm, (1, 1), (1, 1))
        rdms = of.InteractionRDM(opdm, 2 * tpdm)
        return rdms

    def energy_from_opdm(self, opdm_aa: np.ndarray) -> float:
        """Return the energy.

        Args:
            opdm: The alpha-alpha block of the RDM
        """
        rdms = self.rdms_from_opdm_aa(opdm_aa)
        return rdms.expectation(self.hamiltonian).real

    def global_gradient_opdm(self, params: np.ndarray, alpha_opdm: np.ndarray):
        opdm = np.zeros((self.num_qubits, self.num_qubits), dtype=np.complex128)
        opdm[::2, ::2] = alpha_opdm
        opdm[1::2, 1::2] = alpha_opdm
        tpdm = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))

        # now go through and generate all the necessary Z, Y, Y_kl matrices
        kappa_matrix = rhf_params_to_matrix(params,
                                            len(self.occ) + len(self.virt),
                                            self.occ, self.virt)
        kappa_matrix_full = np.kron(kappa_matrix, np.eye(2))
        w_full, v_full = np.linalg.eigh(
            -1j * kappa_matrix_full)  # so that kappa = i U lambda U^
        eigs_scaled_full = get_matrix_of_eigs(w_full)

        grad = np.zeros(self.nocc * self.nvirt, dtype=np.complex128)
        kdelta = np.eye(self.num_qubits)

        # NOW GENERATE ALL TERMS ASSOCIATED WITH THE GRADIENT!!!!!!
        for p in range(self.nocc * self.nvirt):
            grad_params = np.zeros_like(params)
            grad_params[p] = 1
            Y = rhf_params_to_matrix(grad_params,
                                     len(self.occ) + len(self.virt), self.occ,
                                     self.virt)
            Y_full = np.kron(Y, np.eye(2))

            # Now rotate Y into the basis that diagonalizes Z
            Y_kl_full = v_full.conj().T @ Y_full @ v_full
            # now rotate
            # Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
            # into the original basis
            pre_matrix_full = v_full @ (eigs_scaled_full *
                                        Y_kl_full) @ v_full.conj().T

            grad_expectation = -1.0 * np.einsum(
                'ab,pq,aq,pb',
                self.hamiltonian.one_body_tensor,
                pre_matrix_full,
                kdelta,
                opdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ab,pq,bp,aq',
                self.hamiltonian.one_body_tensor,
                pre_matrix_full,
                kdelta,
                opdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ijkl,pq,iq,jpkl',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += -1.0 * np.einsum(
                'ijkl,pq,jq,ipkl',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += -1.0 * np.einsum(
                'ijkl,pq,kp,ijlq',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad_expectation += 1.0 * np.einsum(
                'ijkl,pq,lp,ijkq',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                kdelta,
                tpdm,
                optimize='optimal').real
            grad[p] = grad_expectation

        return grad


def generate_hamiltonian(one_body_integrals: np.ndarray,
                         two_body_integrals: np.ndarray,
                         constant: float,
                         EQ_TOLERANCE: Optional[float] = 1.0E-12):
    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = of.InteractionOperator(constant, one_body_coefficients,
                                                two_body_coefficients)
    return molecular_hamiltonian


def k2_rotgen_grad_one_body(h1, p, q, opdm):
    """
    [h_{pq}, pq]
    """
    expectation = 0.
    # #  (   1.00000) h1(p,a) cre(q) des(a)
    # #  (   1.00000) h1(a,p) cre(a) des(q)
    # expectation += np.dot(h1[p, :], opdm[q, :]) + np.dot(h1[:, p], opdm[:, q])
    # #  (  -1.00000) h1(q,a) cre(p) des(a)
    # #  (  -1.00000) h1(a,q) cre(a) des(p)
    # expectation -= np.dot(h1[q, :], opdm[p, :]) + np.dot(h1[:, q], opdm[:, p])
    # return expectation

    #  (   1.00000) h1(p,a) cre(q) des(a)
    expectation += 1.0 * np.einsum('a,a', h1[p, :], opdm[q, :])
    #  (   1.00000) h1(a,p) cre(a) des(q)
    expectation += 1.0 * np.einsum('a,a', h1[:, p], opdm[:, q])
    #  (  -1.00000) h1(q,a) cre(p) des(a)
    expectation += -1.0 * np.einsum('a,a', h1[q, :], opdm[p, :])
    #  (  -1.00000) h1(a,q) cre(a) des(p)
    expectation += -1.0 * np.einsum('a,a', h1[:, q], opdm[:, p])
    return expectation




def k2_rotgen_grad_fullsym(k2, p, q, tpdm):
    """
    k2-tensor such such that the following terms correspond

    k2[p, q, s, r] p^ q^ r s

    This can probably sped up with blas call to vector dot on reshaped and
    flattened k2 and tpdm
    """
    expectation = 0.
    #  (   4.00000) k2(p,a,b,c) tpdm(q,a,b,c)
    expectation += 4.0 * np.einsum('abc,abc', k2[p, :, :, :], tpdm[q, :, :, :])
    #  (  -4.00000) k2(q,a,b,c) tpdm(p,a,b,c)
    expectation += -4.0 * np.einsum('abc,abc', k2[q, :, :, :], tpdm[p, :, :, :])
    return expectation


def k2_rotgen_grad(k2, p, q, tpdm):
    """
    k2-tensor such such that the following terms correspond

    k2[p, q, s, r] p^ q^ r s

    This can probably sped up with blas call to vector dot on reshaped and
    flattened k2 and tpdm
    """
    expectation = 0.
    #  (  -2.00000) k2(p,a,b,c) cre(q) cre(a) des(b) des(c)
    expectation += -2.0 * np.einsum('abc,abc', k2[p, :, :, :], tpdm[q, :, :, :])
    #  (  -2.00000) k2(p,a,b,c) cre(b) cre(c) des(q) des(a)
    expectation += -2.0 * np.einsum('abc,bca', k2[p, :, :, :], tpdm[:, :, q, :])
    #  (   2.00000) k2(q,a,b,c) cre(p) cre(a) des(b) des(c)
    expectation += 2.0 * np.einsum('abc,abc', k2[q, :, :, :], tpdm[p, :, :, :])
    #  (   2.00000) k2(q,a,b,c) cre(b) cre(c) des(p) des(a)
    expectation += 2.0 * np.einsum('abc,bca', k2[q, :, :, :], tpdm[:, :, p, :])
    return expectation


def k2_rotgen_hessian_fullsym(k2, p, q, r, s, tpdm):
    """
    <| [[k2, p^ q - q^ p], r^s - s^r]|>

    This can be sped up with a dgemm call through numpy
    """
    expectation = 0.
    #  (   4.00000) k2(p,r,a,b) tpdm(q,s,a,b)
    expectation += 4.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[q, s, :, :])
    #  (  -4.00000) k2(p,s,a,b) tpdm(q,r,a,b)
    expectation += -4.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[q, r, :, :])
    #  (   8.00000) k2(p,a,r,b) tpdm(q,a,s,b)
    expectation += 8.0 * np.einsum('ab,ab', k2[p, :, r, :], tpdm[q, :, s, :])
    #  (  -8.00000) k2(p,a,s,b) tpdm(q,a,r,b)
    expectation += -8.0 * np.einsum('ab,ab', k2[p, :, s, :], tpdm[q, :, r, :])
    #  (  -4.00000) k2(q,r,a,b) tpdm(p,s,a,b)
    expectation += -4.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[p, s, :, :])
    #  (   4.00000) k2(q,s,a,b) tpdm(p,r,a,b)
    expectation += 4.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[p, r, :, :])
    #  (  -8.00000) k2(q,a,r,b) tpdm(p,a,s,b)
    expectation += -8.0 * np.einsum('ab,ab', k2[q, :, r, :], tpdm[p, :, s, :])
    #  (   8.00000) k2(q,a,s,b) tpdm(p,a,r,b)
    expectation += 8.0 * np.einsum('ab,ab', k2[q, :, s, :], tpdm[p, :, r, :])
    #  (   4.00000) k2(p,a,b,c) kdelta(q,r) tpdm(s,a,b,c)
    if q == r:
        expectation += 4.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                       tpdm[s, :, :, :])
    #  (  -4.00000) k2(p,a,b,c) kdelta(q,s) tpdm(r,a,b,c)
    if q == s:
        expectation += -4.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                        tpdm[r, :, :, :])
    #  (  -4.00000) k2(q,a,b,c) kdelta(p,r) tpdm(s,a,b,c)
    if p == r:
        expectation += -4.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                        tpdm[s, :, :, :])
    #  (   4.00000) k2(q,a,b,c) kdelta(p,s) tpdm(r,a,b,c)
    if p == s:
        expectation += 4.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                       tpdm[r, :, :, :])
    return expectation


def k2_rotgen_hessian(k2, p, q, r, s, tpdm):
    """
    <| [[k2, p^ q - q^ p], r^s - s^r]|>

    This can be sped up with a dgemm call through numpy
    """
    expectation = 0.
    #  (  -2.00000) k2(p,r,a,b) cre(q) cre(s) des(a) des(b)
    expectation += -2.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[q, s, :, :])
    #  (  -2.00000) k2(p,r,a,b) cre(a) cre(b) des(q) des(s)
    expectation += -2.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[:, :, q, s])
    #  (   2.00000) k2(p,s,a,b) cre(q) cre(r) des(a) des(b)
    expectation += 2.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[q, r, :, :])
    #  (   2.00000) k2(p,s,a,b) cre(a) cre(b) des(q) des(r)
    expectation += 2.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[:, :, q, r])
    #  (  -4.00000) k2(p,a,r,b) cre(q) cre(a) des(s) des(b)
    expectation += -4.0 * np.einsum('ab,ab', k2[p, :, r, :], tpdm[q, :, s, :])
    #  (  -4.00000) k2(p,a,r,b) cre(s) cre(b) des(q) des(a)
    expectation += -4.0 * np.einsum('ab,ba', k2[p, :, r, :], tpdm[s, :, q, :])
    #  (   4.00000) k2(p,a,s,b) cre(q) cre(a) des(r) des(b)
    expectation += 4.0 * np.einsum('ab,ab', k2[p, :, s, :], tpdm[q, :, r, :])
    #  (   4.00000) k2(p,a,s,b) cre(r) cre(b) des(q) des(a)
    expectation += 4.0 * np.einsum('ab,ba', k2[p, :, s, :], tpdm[r, :, q, :])
    #  (   2.00000) k2(q,r,a,b) cre(p) cre(s) des(a) des(b)
    expectation += 2.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[p, s, :, :])
    #  (   2.00000) k2(q,r,a,b) cre(a) cre(b) des(p) des(s)
    expectation += 2.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[:, :, p, s])
    #  (  -2.00000) k2(q,s,a,b) cre(p) cre(r) des(a) des(b)
    expectation += -2.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[p, r, :, :])
    #  (  -2.00000) k2(q,s,a,b) cre(a) cre(b) des(p) des(r)
    expectation += -2.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[:, :, p, r])
    #  (   4.00000) k2(q,a,r,b) cre(p) cre(a) des(s) des(b)
    expectation += 4.0 * np.einsum('ab,ab', k2[q, :, r, :], tpdm[p, :, s, :])
    #  (   4.00000) k2(q,a,r,b) cre(s) cre(b) des(p) des(a)
    expectation += 4.0 * np.einsum('ab,ba', k2[q, :, r, :], tpdm[s, :, p, :])
    #  (  -4.00000) k2(q,a,s,b) cre(p) cre(a) des(r) des(b)
    expectation += -4.0 * np.einsum('ab,ab', k2[q, :, s, :], tpdm[p, :, r, :])
    #  (  -4.00000) k2(q,a,s,b) cre(r) cre(b) des(p) des(a)
    expectation += -4.0 * np.einsum('ab,ba', k2[q, :, s, :], tpdm[r, :, p, :])
    #  (  -2.00000) k2(p,a,b,c) kdelta(q,r) cre(s) cre(a) des(b) des(c)
    if q == r:
        expectation += -2.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                        tpdm[s, :, :, :], optimize=True)
    #  (  -2.00000) k2(p,a,b,c) kdelta(q,r) cre(b) cre(c) des(s) des(a)
    if q == r:
        expectation += -2.0 * np.einsum('abc,bca', k2[p, :, :, :],
                                        tpdm[:, :, s, :], optimize=True)
    #  (   2.00000) k2(p,a,b,c) kdelta(q,s) cre(r) cre(a) des(b) des(c)
    if q == s:
        expectation += 2.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                       tpdm[r, :, :, :], optimize=True)
    #  (   2.00000) k2(p,a,b,c) kdelta(q,s) cre(b) cre(c) des(r) des(a)
    if q == s:
        expectation += 2.0 * np.einsum('abc,bca', k2[p, :, :, :],
                                       tpdm[:, :, r, :], optimize=True)
    #  (   2.00000) k2(q,a,b,c) kdelta(p,r) cre(s) cre(a) des(b) des(c)
    if p == r:
        expectation += 2.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                       tpdm[s, :, :, :], optimize=True)
    #  (   2.00000) k2(q,a,b,c) kdelta(p,r) cre(b) cre(c) des(s) des(a)
    if p == r:
        expectation += 2.0 * np.einsum('abc,bca', k2[q, :, :, :],
                                       tpdm[:, :, s, :], optimize=True)
    #  (  -2.00000) k2(q,a,b,c) kdelta(p,s) cre(r) cre(a) des(b) des(c)
    if p == s:
        expectation += -2.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                        tpdm[r, :, :, :], optimize=True)
    #  (  -2.00000) k2(q,a,b,c) kdelta(p,s) cre(b) cre(c) des(r) des(a)
    if p == s:
        expectation += -2.0 * np.einsum('abc,bca', k2[q, :, :, :],
                                        tpdm[:, :, r, :], optimize=True)
    return expectation


def k2_rotgen_hess_one_body(h1, p, q, r, s, opdm):
    expectation = 0.
    #  (   1.00000) h1(p,r) cre(q) des(s)
    expectation += 1.0 * opdm[q, s] * h1[p, r]
    #  (  -1.00000) h1(p,s) cre(q) des(r)
    expectation += -1.0 * opdm[q, r] * h1[p, s]
    #  (  -1.00000) h1(q,r) cre(p) des(s)
    expectation += -1.0 * opdm[p, s] * h1[q, r]
    #  (   1.00000) h1(q,s) cre(p) des(r)
    expectation += 1.0 * opdm[p, r] * h1[q, s]
    #  (   1.00000) h1(r,p) cre(s) des(q)
    expectation += 1.0 * opdm[s, q] * h1[r, p]
    #  (  -1.00000) h1(r,q) cre(s) des(p)
    expectation += -1.0 * opdm[s, p] * h1[r, q]
    #  (  -1.00000) h1(s,p) cre(r) des(q)
    expectation += -1.0 * opdm[r, q] * h1[s, p]
    #  (   1.00000) h1(s,q) cre(r) des(p)
    expectation += 1.0 * opdm[r, p] * h1[s, q]

    #  (   1.00000) h1(p,a) kdelta(q,r) cre(s) des(a)
    if q == r:
        expectation += 1.0 * np.einsum('a,a', h1[p, :], opdm[s, :])
    #  (  -1.00000) h1(p,a) kdelta(q,s) cre(r) des(a)
    if q == s:
        expectation += -1.0 * np.einsum('a,a', h1[p, :], opdm[r, :])
    #  (  -1.00000) h1(q,a) kdelta(p,r) cre(s) des(a)
    if p == r:
        expectation += -1.0 * np.einsum('a,a', h1[q, :], opdm[s, :])
    #  (   1.00000) h1(q,a) kdelta(p,s) cre(r) des(a)
    if p == s:
        expectation += 1.0 * np.einsum('a,a', h1[q, :], opdm[r, :])
    #  (   1.00000) h1(a,p) kdelta(q,r) cre(a) des(s)
    if q == r:
        expectation += 1.0 * np.einsum('a,a', h1[:, p], opdm[:, s])
    #  (  -1.00000) h1(a,p) kdelta(q,s) cre(a) des(r)
    if q == s:
        expectation += -1.0 * np.einsum('a,a', h1[:, p], opdm[:, r])
    #  (  -1.00000) h1(a,q) kdelta(p,r) cre(a) des(s)
    if p == r:
        expectation += -1.0 * np.einsum('a,a', h1[:, q], opdm[:, s])
    #  (   1.00000) h1(a,q) kdelta(p,s) cre(a) des(r)
    if p == s:
        expectation += 1.0 * np.einsum('a,a', h1[:, q], opdm[:, r])
    return expectation


def orbital_optimize_spinorb(reduced_ham: of.InteractionOperator,
                             tpdm: np.ndarray, grad_eps=1.0E-6, maxiter=300,
                             verbose: bool=False) -> np.ndarray:
    """
    Implement orbital rotation on reduced Hamiltonian

    orbital rotation is implemented by optimzing

    .. math:
        min_{\kappa} <\psi|U(\kappa)^{\dagger} H U(\kappa)|\psi>

    for the $\kappa = \sum_{p<q} k_{pq}E_{pq} - E_{qp}

    where E_{pq} = p_{a}^q_{a} + p_{b}^q_{b}
    which is the sum of different spin spaces.

    :param reduced_ham: two-body only Hamiltonian.  This can be constructed
                        with openfermion.make_reduced_ham. It must be
                        antisymmetric with respect to spin orbs so make
                        the reduced ham with an anitsymmeterized 2-electron
                        integrals.
                        soei, stei = spinorb_from_spatial(obi, tbi)
                        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
                        io_op = of.InteractionOperator(0, soei, 0.25 * astei)
                        rham = of.make_reduced_hamiltonian(io_op, ...)
    :param tpdm: tpdm as a 4-tensor in spin-orbital basis as p^ q^ r s
                 <1'2'|21> as the electron order.
    :return: basis unitary that implementes the energy minimizing rotation
    """
    rham = copy.deepcopy(reduced_ham)
    grad_residual = np.infty
    norbs = int(tpdm.shape[0]) // 2  # number of spatial orbs
    # lower-triangle indices in variable order
    ridx, cidx = np.tril_indices(norbs, -1)  # row-major iteration
    num_vars = norbs * (norbs - 1) // 2
    delta = 0.05

    if verbose:
        print("Initial Energy")
        print(np.einsum('ijkl,ijkl', rham.two_body_tensor, tpdm))
        print("Entering Newton Steps:")
    orb_rotations = []
    current_iter = 0
    while grad_residual > grad_eps and current_iter < maxiter:
        comm_grad = np.zeros(num_vars)
        k2_tensor = rham.two_body_tensor.real.transpose(0, 1, 3, 2)
        for idx, (v, o) in enumerate(zip(ridx, cidx)):
            comm_grad_val = k2_rotgen_grad_fullsym(k2_tensor, 2 * v, 2 * o, tpdm) + \
                            k2_rotgen_grad_fullsym(k2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            comm_grad[idx] = comm_grad_val

        grad_residual = np.linalg.norm(comm_grad)

        comm_hess = np.zeros((num_vars, num_vars))
        for idx, (v1, o1) in enumerate(zip(ridx, cidx)):
            for jdx, (v2, o2) in enumerate(zip(ridx, cidx)):
                if idx >= jdx:
                    hess_val = 0.
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian_fullsym(
                            k2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian_fullsym(
                            k2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )

                    comm_hess[idx, jdx] = 0.5 * hess_val
                    comm_hess[jdx, idx] = 0.5 * hess_val

        assert np.allclose(comm_hess, comm_hess.T)

        new_fr_vals = -np.linalg.pinv(comm_hess) @ comm_grad

        new_kappa = np.zeros((norbs, norbs))
        new_kappa[ridx, cidx] = new_fr_vals
        new_kappa -= new_kappa.T  # upper triangle has negative signs

        basis_change = expm(new_kappa)
        rham.rotate_basis(basis_change)

        orb_rotations.append(basis_change)
        current_iter += 1

        if verbose:
            print("\t{}\t{:5.10f}\t{:5.10f}".format(
                current_iter,
                np.einsum('ijkl,ijkl', rham.two_body_tensor, tpdm),
                grad_residual)
            )

    # no list reversal required because
    # e^{-KN}...e^{-K2}e^{-K1}He^{K1}e^{K2}e^{K3}...e^{KN}
    # thus total rotation is e^{K1}e^{K2}...e^{KN} which we can implement we
    # just normal reduce
    return reduce(np.dot, orb_rotations)


def oo_so_sep(oei: np.ndarray, tei: np.ndarray, tpdm: np.ndarray,
              grad_eps=1.0E-6, maxiter=300, verbose: bool=False) -> np.ndarray:
    """
    Implement orbital rotation on reduced Hamiltonian

    orbital rotation is implemented by optimzing

    .. math:
        min_{\kappa} <\psi|U(\kappa)^{\dagger} H U(\kappa)|\psi>

    for the $\kappa = \sum_{p<q} k_{pq}E_{pq} - E_{qp}

    where E_{pq} = p_{a}^q_{a} + p_{b}^q_{b}
    which is the sum of different spin spaces.

    :param reduced_ham: two-body only Hamiltonian.  This can be constructed
                        with openfermion.make_reduced_ham. It must be
                        antisymmetric with respect to spin orbs so make
                        the reduced ham with an anitsymmeterized 2-electron
                        integrals.
                        soei, stei = spinorb_from_spatial(obi, tbi)
                        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
                        io_op = of.InteractionOperator(0, soei, 0.25 * astei)
                        rham = of.make_reduced_hamiltonian(io_op, ...)
    :param tpdm: tpdm as a 4-tensor in spin-orbital basis as p^ q^ r s
                 <1'2'|21> as the electron order.
    :return: basis unitary that implementes the energy minimizing rotation
    """
    # Get number of electrons and 1-RDM from 2-RDM
    c = -np.einsum('ijji', tpdm)
    a = 1
    b = -1
    roots = [(-b + np.sqrt(b**2 - (4 * a * c))) / (2 * a),
             (-b - np.sqrt(b**2 - (4 * a * c))) / (2 * a)]
    n_elec = int(np.max(roots))
    opdm = of.map_two_pdm_to_one_pdm(tpdm, n_elec)

    # Set of antisymmeterized spin-orb Hamiltonian
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molham = of.InteractionOperator(0, soei, 0.25 * astei)
    molham1 = of.InteractionOperator(0, soei, np.zeros_like(astei))
    molham2 = of.InteractionOperator(0, np.zeros_like(soei), 0.25 * astei)
    rham = of.make_reduced_hamiltonian(molham, n_elec)

    # set up optimization requirements
    grad_residual = np.infty
    norbs = int(tpdm.shape[0]) // 2  # number of spatial orbs
    # lower-triangle indices in variable order
    ridx, cidx = np.tril_indices(norbs, -1)  # row-major iteration
    num_vars = norbs * (norbs - 1) // 2
    delta = 0.05


    wf = of.jw_hartree_fock_state(n_elec, 2 * norbs)
    up_op = [of.FermionOperator(((xx, 1))) for xx in range(2 * norbs)]
    up_fop = [of.FermionOperator(((xx, 1))) for xx in range(2 * norbs)]
    up_op = [of.get_sparse_operator(op, n_qubits=2 * norbs) for op in up_op]
    dwn_op = [op.conj().T for op in up_op]
    dwn_fop = [of.hermitian_conjugated(op) for op in up_fop]

    if verbose:
        print("Initial Energy")
        print(np.einsum('ijkl,ijkl', molham.two_body_tensor.real, tpdm) +
              np.einsum('ij,ij', molham.one_body_tensor.real, opdm))
        print("Entering Newton Steps:")
    orb_rotations = []
    current_iter = 0
    while grad_residual > grad_eps and current_iter < maxiter:
        comm_grad = np.zeros(num_vars)

        h_mat = of.get_sparse_operator(molham)
        h1_mat = of.get_sparse_operator(molham1)
        v2_mat = of.get_sparse_operator(molham2)
        v2_tensor = molham2.two_body_tensor.real.transpose(0, 1, 3, 2)
        h1_tensor = molham1.one_body_tensor.real
        k2_tensor = rham.two_body_tensor.real.transpose(0, 1, 3, 2)

        for idx, (v, o) in enumerate(zip(ridx, cidx)):
            comm_grad_val = k2_rotgen_grad_fullsym(v2_tensor, 2 * v, 2 * o, tpdm) + \
                            k2_rotgen_grad_fullsym(v2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            comm_grad_val += k2_rotgen_grad_one_body(h1_tensor, 2 * v, 2 * o, opdm) + \
                             k2_rotgen_grad_one_body(h1_tensor, 2 * v + 1, 2 * o + 1,
                                                     opdm)
            comm_grad[idx] = comm_grad_val
            test_val = k2_rotgen_grad_fullsym(k2_tensor, 2 * v, 2 * o, tpdm) + \
                       k2_rotgen_grad_fullsym(k2_tensor, 2 * v + 1, 2 * o + 1, tpdm)

            assert np.isclose(test_val, comm_grad_val)

        grad_residual = np.linalg.norm(comm_grad)

        comm_hess = np.zeros((num_vars, num_vars))
        for idx, (v1, o1) in enumerate(zip(ridx, cidx)):
            for jdx, (v2, o2) in enumerate(zip(ridx, cidx)):
                if idx >= jdx:
                    hess_val = 0.
                    for sigma, tau in product(range(2), repeat=2):
                        tb_hess_val = k2_rotgen_hessian(
                            v2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                        ob_hess_val = k2_rotgen_hess_one_body(
                            h1_tensor,
                            2 * v1 + sigma, 2 * o1 + sigma,
                            2 * v2 + tau, 2 * o2 + tau,
                            opdm)
                        hess_val += ob_hess_val
                        hess_val += tb_hess_val

                        # print("Sigma Tau ", sigma, tau)
                        fop_inner = up_op[2 * v1 + sigma] @ dwn_op[2 * o1 + sigma] - \
                                    up_op[2 * o1 + sigma] @ dwn_op[2 * v1 + sigma]
                        fop_outer = up_op[2 * v2 + tau] @ dwn_op[2 * o2 + tau] - \
                                    up_op[2 * o2 + tau] @ dwn_op[2 * v2 + tau]
                        tval = (wf.T @ of.commutator(of.commutator(h1_mat, fop_inner), fop_outer) @ wf).real
                        # print("cirq compare one-body ", tval, ob_hess_val)
                        assert np.isclose(tval, ob_hess_val)
                        tval = (wf.T @ of.commutator(of.commutator(v2_mat, fop_inner), fop_outer) @ wf).real
                        # print("cirq compare two-body", tval, tb_hess_val)
                        assert np.isclose(tval, tb_hess_val)

                    for sigma, tau in product(range(2), repeat=2):
                        tb_hess_val = k2_rotgen_hessian(
                            v2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )
                        ob_hess_val = k2_rotgen_hess_one_body(
                            h1_tensor,
                            2 * v2 + sigma, 2 * o2 + sigma,
                            2 * v1 + tau, 2 * o1 + tau,
                            opdm)
                        hess_val += ob_hess_val
                        hess_val += tb_hess_val

                        # print("Sigma Tau ", sigma, tau)
                        fop_inner = up_op[2 * v2 + sigma] @ dwn_op[2 * o2 + sigma] - \
                                    up_op[2 * o2 + sigma] @ dwn_op[2 * v2 + sigma]
                        fop_outer = up_op[2 * v1 + tau] @ dwn_op[2 * o1 + tau] - \
                                    up_op[2 * o1 + tau] @ dwn_op[2 * v1 + tau]
                        tval = (wf.T @ of.commutator(of.commutator(h1_mat, fop_inner), fop_outer) @ wf).real
                        # print("cirq compare one-body ", tval, ob_hess_val)
                        assert np.isclose(tval, ob_hess_val)
                        tval = (wf.T @ of.commutator(of.commutator(v2_mat, fop_inner), fop_outer) @ wf).real
                        # print("cirq compare two-body", tval, tb_hess_val)
                        assert np.isclose(tval, tb_hess_val)

                    comm_hess[idx, jdx] = 0.5 * hess_val
                    comm_hess[jdx, idx] = 0.5 * hess_val

                    test_hess_val = 0.
                    for sigma, tau in product(range(2), repeat=2):
                        test_hess_val += k2_rotgen_hessian(
                            k2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                    for sigma, tau in product(range(2), repeat=2):
                        test_hess_val += k2_rotgen_hessian(
                            k2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )

                    # print(test_hess_val, hess_val)
                    assert np.isclose(test_hess_val, hess_val)

        assert np.allclose(comm_hess, comm_hess.T)

        new_fr_vals = -np.linalg.pinv(comm_hess) @ comm_grad

        new_kappa = np.zeros((norbs, norbs))
        new_kappa[ridx, cidx] = new_fr_vals
        new_kappa -= new_kappa.T  # upper triangle has negative signs

        basis_change = expm(new_kappa)
        rham.rotate_basis(basis_change)
        molham.rotate_basis(basis_change)

        orb_rotations.append(basis_change)
        current_iter += 1

        if verbose:
            print("\t{}\t{:5.10f}\t{:5.10f}".format(
                current_iter,
                np.einsum('ijkl,ijkl', rham.two_body_tensor, tpdm),
                grad_residual)
            )

    # no list reversal required because
    # e^{-KN}...e^{-K2}e^{-K1}He^{K1}e^{K2}e^{K3}...e^{KN}
    # thus total rotation is e^{K1}e^{K2}...e^{KN} which we can implement we
    # just normal reduce
    return reduce(np.dot, orb_rotations)


def main():
    np.set_printoptions(linewidth=400)
    molecule = load_hydrogen_fluoride_molecule(1.7)
    norbs = molecule.n_orbitals
    hf_opdm = np.diag([1] * molecule.n_electrons + [0] * (2 * molecule.n_orbitals - molecule.n_electrons))
    hf_tpdm = 2 * of.wedge(hf_opdm, hf_opdm, (1, 1), (1, 1)).real

    pyscf_scf =  molecule._pyscf_data['scf']
    pyscf_molecule = molecule._pyscf_data['mol']
    from pyscf import ao2mo

    S = pyscf_scf.get_ovlp()
    Hcore = pyscf_scf.get_hcore()
    C = molecule.canonical_orbitals
    assert np.allclose(C.T @ S @ C, np.eye(C.shape[0]))

    # Ritate back to AO basis
    OEI = of.general_basis_change(molecule.one_body_integrals, C.T @ S, (1, 0))
    assert np.allclose(OEI, Hcore)

    two_electron_compressed = ao2mo.kernel(pyscf_molecule, C)
    two_electron_integrals = ao2mo.restore(
        1,  # no permutation symmetry
        two_electron_compressed, molecule.n_orbitals)

    ao_eri = pyscf_molecule.intor('int2e', aosym='s1')  # (ij|kl)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    TEI_test = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')
    assert np.allclose(TEI_test, molecule.two_body_integrals)

    ao_TEI = of.general_basis_change(molecule.two_body_integrals, C.T @ S,
                                     (1, 1, 0, 0)).transpose(0, 3, 1, 2)
    assert np.allclose(ao_TEI, ao_eri)


    _, X = sp.linalg.eigh(Hcore, S)
    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(ao_eri, X, (1, 0, 1, 0)))
    molecular_hamiltonian = generate_hamiltonian(obi, tbi,
                                                 molecule.nuclear_repulsion)
    fqe_ham = RestrictedHamiltonian((obi, np.einsum("ijlk", -0.5 * tbi)))

    soei, stei = spinorb_from_spatial(obi, tbi)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.make_reduced_hamiltonian(molecular_hamiltonian,
                                              molecule.n_electrons)

    rhf_obj = RestrictedHartreeFockObjective(molecular_hamiltonian,
                                             molecule.n_electrons)
    rhf_obj_rham = RestrictedHartreeFockObjective(reduced_ham,
                                             molecule.n_electrons)

    print("mol hf ", molecule.hf_energy - molecule.nuclear_repulsion)
    # unitary = orbital_optimize_spinorb(reduced_ham=copy.deepcopy(reduced_ham),
    #                          tpdm=hf_tpdm, verbose=True)
    unitary = oo_so_sep(obi, tbi, hf_tpdm, verbose=True)
    print("molecule hf ", molecule.hf_energy - molecule.nuclear_repulsion)

    reduced_ham.rotate_basis(unitary)
    print(np.einsum('ijkl,ijkl', reduced_ham.two_body_tensor.real, hf_tpdm))
    exit()


    init_guess = np.zeros(rhf_obj.nocc * rhf_obj.nvirt)
    occ = rhf_obj.occ
    virt = rhf_obj.virt
    nocc = len(occ)
    nvirt = len(virt)
    grad_residual = np.infty

    k2_tensor = reduced_ham.two_body_tensor.real.transpose(0, 1, 3, 2)
    # wf = of.jw_hartree_fock_state(molecule.n_electrons, molecule.n_qubits)
    # k2_mat = of.get_sparse_operator(of.get_fermion_operator(reduced_ham))
    # h_mat = of.get_sparse_operator(molecular_hamiltonian)
    # print(wf.T @ k2_mat @ wf)
    print(rhf_obj_rham.energy_from_opdm(hf_opdm[::2, ::2]))

    # up_op = [of.FermionOperator(((xx, 1))) for xx in range(molecule.n_qubits)]
    # up_fop = [of.FermionOperator(((xx, 1))) for xx in range(molecule.n_qubits)]
    # up_op = [of.get_sparse_operator(op, n_qubits=molecule.n_qubits) for op in up_op]
    # dwn_op = [op.conj().T for op in up_op]
    # dwn_fop = [of.hermitian_conjugated(op) for op in up_fop]
    delta = 0.02
    while grad_residual > 1.0E-6:
        comm_grad = np.zeros((nocc * nvirt))
        for idx, (v, o) in enumerate(product(virt, occ)):
            # print("V", v, "O", o)
            # print("{}^ {} - {}^ {}".format(v, o, o, v))
            comm_grad_val = k2_rotgen_grad(k2_tensor, 2 * v, 2 * o, hf_tpdm) + \
                            k2_rotgen_grad(k2_tensor, 2 * v + 1, 2 * o + 1, hf_tpdm)
            comm_grad[idx] = comm_grad_val
            comm_grad_val_test = k2_rotgen_grad_fullsym(k2_tensor, 2 * v, 2 * o, hf_tpdm.transpose(0, 1, 3, 2)) + \
                            k2_rotgen_grad_fullsym(k2_tensor, 2 * v + 1, 2 * o + 1, hf_tpdm.transpose(0, 1, 3, 2))
            assert np.allclose(comm_grad_val_test, comm_grad_val)
            # print(comm_grad[idx])
            # init_guess = np.zeros(rhf_obj.nocc * rhf_obj.nvirt)
            # init_guess[idx] = 1.0E-4
            # kappa = rhf_params_to_matrix(
            #     init_guess, len(occ) + len(virt), occ, virt)
            # # print(kappa)
            # fop = up_op[2 * v] @ dwn_op[2 * o] - up_op[2 * o] @ dwn_op[2 * v]
            # fermion_op = up_fop[2 * v] * dwn_fop[2 * o] - up_fop[2 * o] * dwn_fop[2 * v]
            # # print(of.get_interaction_operator(fermion_op).one_body_tensor[::2, ::2] * 1.0E-4)
            # fop += up_op[2 * v + 1] @ dwn_op[2 * o + 1] - up_op[2 * o + 1] @ dwn_op[2 * v + 1]

            # print((wf.T @ of.commutator(h_mat, fop) @ wf).real)

        # for idx, (v, o) in enumerate(product(virt, occ)):
        #     print(idx, v, o, comm_grad[idx])
        # print(comm_grad)
        # exit()
        grad_residual = np.linalg.norm(comm_grad)

        # g_hf = rhf_obj.global_gradient_opdm(init_guess, alpha_opdm=hf_opdm[::2, ::2])
        # print(g_hf.real)

        # g_hfrh = rhf_obj_rham.global_gradient_opdm(init_guess, alpha_opdm=hf_opdm[::2, ::2])
        # print(g_hfrh.real)

        # # Finite difference
        # grad_fd = np.zeros((len(occ) * len(virt)))
        # for idx, (v, o) in enumerate(product(virt, occ)):
        #     init_guess = np.zeros(rhf_obj.nocc * rhf_obj.nvirt)
        #     init_guess[idx] = 1.0E-4
        #     kappa = rhf_params_to_matrix(
        #         init_guess, len(occ) + len(virt), occ, virt)
        #     u = sp.linalg.expm(kappa)
        #     new_opdm = u @ hf_opdm[::2, ::2] @ u.T
        #     plus_energy = rhf_obj_rham.energy_from_opdm(new_opdm)

        #     init_guess = np.zeros(rhf_obj.nocc * rhf_obj.nvirt)
        #     init_guess[idx] = -1.0E-4
        #     kappa = rhf_params_to_matrix(
        #         init_guess, len(occ) + len(virt), occ, virt)
        #     u = sp.linalg.expm(kappa)
        #     new_opdm = u @ hf_opdm[::2, ::2] @ u.T
        #     minus_energy = rhf_obj_rham.energy_from_opdm(new_opdm)

        #     grad_fd[idx] =  ((plus_energy.real - minus_energy.real) / (2 * 1.0E-4))
        # print(grad_fd.real)

        comm_hess = np.zeros((nocc * nvirt, nocc * nvirt))
        for idx, (v1, o1) in enumerate(product(virt, occ)):
            for jdx, (v2, o2) in enumerate(product(virt, occ)):
                if idx >= jdx:
                    # print("V1", v1, "O1", o1, "V2", v2, "O2", o2)
                    # print("{}^ {} - {}^ {}".format(v1, o1, o1, v1),
                    #       "{}^ {} - {}^ {}".format(v2, o2, o2, v2)
                    #       )
                    # fop_inner = up_op[2 * v1] @ dwn_op[2 * o1] - \
                    #             up_op[2 * o1] @ dwn_op[2 * v1]
                    # fop_inner += up_op[2 * v1 + 1] @ dwn_op[2 * o1 + 1] - \
                    #              up_op[2 * o1 + 1] @ dwn_op[2 * v1 + 1]
                    # fop_outer = up_op[2 * v2] @ dwn_op[2 * o2] - \
                    #             up_op[2 * o2] @ dwn_op[2 * v2]
                    # fop_outer += up_op[2 * v2 + 1] @ dwn_op[2 * o2 + 1] - \
                    #              up_op[2 * o2 + 1] @ dwn_op[2 * v2 + 1]

                    # print((wf.T @ of.commutator(of.commutator(h_mat, fop_inner),
                    #                             fop_outer) @ wf).real)
                    hess_val = k2_rotgen_hessian(k2_tensor,
                                                 2 * v1, 2 * o1, 2 * v2, 2 * o2,
                                                 hf_tpdm) + \
                               k2_rotgen_hessian(k2_tensor,
                                                 2 * v1, 2 * o1, 2 * v2 + 1, 2 * o2 + 1,
                                                 hf_tpdm) + \
                               k2_rotgen_hessian(k2_tensor,
                                                 2 * v1 + 1, 2 * o1 + 1, 2 * v2, 2 * o2,
                                                 hf_tpdm) + \
                               k2_rotgen_hessian(k2_tensor,
                                                 2 * v1 + 1, 2 * o1 + 1, 2 * v2 + 1, 2 * o2 + 1,
                                                 hf_tpdm)

                    hess_val_test = k2_rotgen_hessian_fullsym(k2_tensor,
                                                 2 * v1, 2 * o1, 2 * v2, 2 * o2,
                                                 hf_tpdm.transpose(0, 1, 3, 2)) + \
                               k2_rotgen_hessian_fullsym(k2_tensor,
                                                 2 * v1, 2 * o1, 2 * v2 + 1,
                                                 2 * o2 + 1,
                                                 hf_tpdm.transpose(0, 1, 3, 2)) + \
                               k2_rotgen_hessian_fullsym(k2_tensor,
                                                 2 * v1 + 1, 2 * o1 + 1, 2 * v2,
                                                 2 * o2,
                                                 hf_tpdm.transpose(0, 1, 3, 2)) + \
                               k2_rotgen_hessian_fullsym(k2_tensor,
                                                 2 * v1 + 1, 2 * o1 + 1,
                                                 2 * v2 + 1, 2 * o2 + 1,
                                                 hf_tpdm.transpose(0, 1, 3, 2))
                    assert np.isclose(hess_val, hess_val_test)
                    comm_hess[idx, jdx] = hess_val
                    comm_hess[jdx,  idx] = hess_val
                    # print(hess_val)

        # build augmented Hessian
        dvec = comm_grad.reshape((-1, 1))
        aug_hess = np.hstack((np.array([[0]]), dvec.T))
        aug_hess = np.vstack((aug_hess, np.hstack((dvec, comm_hess))))

        w, v = np.linalg.eig(aug_hess)
        sort_idx = np.argsort(w)
        w = w[sort_idx]
        v = v[:, sort_idx]
        if np.abs(v[0, 0]) >= 1.0E-13:
            new_fr_vals = v[1:, [0]].flatten() / v[0, 0]
        else:
            new_fr_vals = v[1:, [0]].flatten()

        # if np.max(abs(new_fr_vals)) >= delta:
        #     new_fr_vals = delta * new_fr_vals / np.max(abs(new_fr_vals))


        new_kappa = rhf_params_to_matrix(new_fr_vals, norbs,
                                         occ, virt)
        print(new_kappa)
        np.save("/Users/nickrubin/dev/qcpanop/qcpanop/scf/restricted_new_kappa.npy", new_kappa)
        exit()
        basis_change = expm(new_kappa)

        rdms = of.InteractionRDM(hf_opdm, hf_tpdm)
        # print(rdms.expectation(reduced_ham) + molecule.nuclear_repulsion)
        reduced_ham.rotate_basis(basis_change)
        print(rdms.expectation(reduced_ham))#  + molecule.nuclear_repulsion)
        k2_tensor = reduced_ham.two_body_tensor.transpose(0, 1, 3, 2)

    print("hf-scf energy ", molecule.hf_energy - molecule.nuclear_repulsion)

    reduced_ham = of.make_reduced_hamiltonian(molecular_hamiltonian,
                                              molecule.n_electrons)



if __name__ == "__main__":
    main()