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

from qcpanop.scf.commutators import (k2_rotgen_grad, k2_rotgen_hessian,
                                     k2_rotgen_grad_one_body,
                                     k2_rotgen_hess_one_body,
                                     spinless_rotgrad_twobody,
                                     spinless_rotgrad_onebody,
                                     spinless_rothess_onebody,
                                     spinless_rothess_twobody)


class Stepper:

    def __init__(self, step_type):
        """Define which type of step to take with hess and grad info"""
        self.step_type = step_type
        self.delta = None

    def __call__(self, grad, hess):
        if self.step_type == 'newton-raphson':
            return self._nr_step(grad, hess)
        elif self.step_type == 'augmented-hessian':
            return self._ah_step(grad, hess)
        elif self.step_type == 'diagonal-hessian':
            return self._diagonal_hess(grad, hess)
        else:
            raise ValueError("Unrecognized step type")

    def _nr_step(self, grad, hess):
        return -np.linalg.pinv(hess) @ grad

    def _ah_step(self, grad, hess):
        dvec = grad.reshape((-1, 1))
        aug_hess = np.hstack((np.array([[0]]), dvec.T))
        aug_hess = np.vstack((aug_hess, np.hstack((dvec, hess))))

        w, v = np.linalg.eig(aug_hess)
        sort_idx = np.argsort(w)
        w = w[sort_idx]
        v = v[:, sort_idx]
        if np.abs(v[0, 0]) >= 1.0E-13:
            new_fr_vals = v[1:, [0]].flatten() / v[0, 0]
        else:
            new_fr_vals = v[1:, [0]].flatten()

        if self.delta is not None:
            if np.max(abs(new_fr_vals)) >= self.delta:
                new_fr_vals = self.delta * new_fr_vals / np.max(abs(new_fr_vals))
            return new_fr_vals
        else:
            return new_fr_vals

    def _diagonal_hess(self, grad, hess):
        return -np.linalg.pinv(np.diag(np.diagonal(hess))) @ grad



def orbital_optimize_spinorb(reduced_ham: of.InteractionOperator,
                             tpdm: np.ndarray, grad_eps=1.0E-6, maxiter=300,
                             verbose: bool=False, method='newton-raphson') -> np.ndarray:
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

    stepper = Stepper(method)

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
            comm_grad_val = k2_rotgen_grad(k2_tensor, 2 * v, 2 * o, tpdm) + \
                            k2_rotgen_grad(k2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            comm_grad[idx] = comm_grad_val

        grad_residual = np.linalg.norm(comm_grad)

        comm_hess = np.zeros((num_vars, num_vars))
        for idx, (v1, o1) in enumerate(zip(ridx, cidx)):
            for jdx, (v2, o2) in enumerate(zip(ridx, cidx)):
                if idx >= jdx:
                    hess_val = 0.
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian(
                            k2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian(
                            k2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )

                    comm_hess[idx, jdx] = 0.5 * hess_val
                    comm_hess[jdx, idx] = 0.5 * hess_val

        assert np.allclose(comm_hess, comm_hess.T)

        new_fr_vals = stepper(comm_grad, comm_hess)

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


def get_nelectrons_from_tpdm(tpdm):
    c = -np.einsum('ijji', tpdm)
    a = 1
    b = -1
    roots = [(-b + np.sqrt(b**2 - (4 * a * c))) / (2 * a),
             (-b - np.sqrt(b**2 - (4 * a * c))) / (2 * a)]
    n_elec = int(np.max(roots))
    return n_elec


def oo_so_sep(obi: np.ndarray, tbi: np.ndarray,
              tpdm: np.ndarray, grad_eps=1.0E-6, maxiter=300,
              verbose: bool=False, method='newton-raphson') -> np.ndarray:
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
    nelec = get_nelectrons_from_tpdm(tpdm)
    opdm = of.map_two_pdm_to_one_pdm(tpdm, nelec)
    soei, stei = spinorb_from_spatial(obi, tbi)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    moleham = of.InteractionOperator(0, soei, 0.25 * astei)
    redham = of.make_reduced_hamiltonian(moleham, n_electrons=nelec)
    # k2_tensor = redham.two_body_tensor.transpose(0, 1, 3, 2)
    #2_tensor = moleham.two_body_tensor.transpose(0, 1, 3, 2)
    # h1_tensor = moleham.one_body_tensor
    grad_residual = np.infty
    norbs = int(tpdm.shape[0]) // 2  # number of spatial orbs
    # lower-triangle indices in variable order
    ridx, cidx = np.tril_indices(norbs, -1)  # row-major iteration
    num_vars = norbs * (norbs - 1) // 2
    delta = 0.05

    stepper = Stepper(method)

    if verbose:
        print("Initial Energy")
        print(np.einsum('ijkl,ijkl', redham.two_body_tensor, tpdm))
        print("Entering Newton Steps:")
    orb_rotations = []
    current_iter = 0
    while grad_residual > grad_eps and current_iter < maxiter:
        comm_grad = np.zeros(num_vars)
        k2_tensor = redham.two_body_tensor.real.transpose(0, 1, 3, 2)
        v2_tensor = moleham.two_body_tensor.transpose(0, 1, 3, 2)
        h1_tensor = moleham.one_body_tensor
        for idx, (v, o) in enumerate(zip(ridx, cidx)):
            comm_grad_val = k2_rotgen_grad(v2_tensor, 2 * v, 2 * o, tpdm) + \
                            k2_rotgen_grad(v2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            comm_grad_val += k2_rotgen_grad_one_body(h1_tensor, 2 * v, 2 * o,
                                                     opdm) + \
                             k2_rotgen_grad_one_body(h1_tensor, 2 * v + 1,
                                                     2 * o + 1, opdm)
            test_comm_grad_val = k2_rotgen_grad(k2_tensor, 2 * v, 2 * o, tpdm) + \
                            k2_rotgen_grad(k2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            assert np.isclose(test_comm_grad_val, comm_grad_val)
            comm_grad[idx] = comm_grad_val

        grad_residual = np.linalg.norm(comm_grad)

        comm_hess = np.zeros((num_vars, num_vars))
        for idx, (v1, o1) in enumerate(zip(ridx, cidx)):
            for jdx, (v2, o2) in enumerate(zip(ridx, cidx)):
                if idx >= jdx:
                    hess_val = 0.
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian(
                            v2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                        hess_val += k2_rotgen_hess_one_body(h1_tensor,
                                                            2 * v1 + sigma,
                                                            2 * o1 + sigma,
                                                            2 * v2 + tau,
                                                            2 * o2 + tau, opdm)
                    for sigma, tau in product(range(2), repeat=2):
                        hess_val += k2_rotgen_hessian(
                            v2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )
                        hess_val += k2_rotgen_hess_one_body(h1_tensor,
                                                            2 * v2 + sigma,
                                                            2 * o2 + sigma,
                                                            2 * v1 + tau,
                                                            2 * o1 + tau, opdm)

                    comm_hess[idx, jdx] = 0.5 * hess_val
                    comm_hess[jdx, idx] = 0.5 * hess_val

        assert np.allclose(comm_hess, comm_hess.T)

        new_fr_vals = stepper(comm_grad, comm_hess)

        new_kappa = np.zeros((norbs, norbs))
        new_kappa[ridx, cidx] = new_fr_vals
        new_kappa -= new_kappa.T  # upper triangle has negative signs

        basis_change = expm(new_kappa)
        redham.rotate_basis(basis_change)
        moleham.rotate_basis(basis_change)

        orb_rotations.append(basis_change)
        current_iter += 1

        if verbose:
            print("\t{}\t{:5.10f}\t{:5.10f}".format(
                current_iter,
                np.einsum('ijkl,ijkl', redham.two_body_tensor, tpdm),
                grad_residual)
            )

    # no list reversal required because
    # e^{-KN}...e^{-K2}e^{-K1}He^{K1}e^{K2}e^{K3}...e^{KN}
    # thus total rotation is e^{K1}e^{K2}...e^{KN} which we can implement we
    # just normal reduce
    return reduce(np.dot, orb_rotations)


def spatial_oo(oei: np.ndarray, tei: np.ndarray, sopdm: np.ndarray,
               stpdm: np.ndarray, tpdm=None, opdm=None, grad_eps=1.0E-6, maxiter=300,
               verbose: bool = False, method='newton-raphson') -> np.ndarray:
    """
    True orbital optimization using spin-summed opdm
    """
    grad_residual = np.infty
    norbs = int(tpdm.shape[0]) // 2  # number of spatial orbs
    # lower-triangle indices in variable order
    ridx, cidx = np.tril_indices(norbs, -1)  # row-major iteration
    num_vars = norbs * (norbs - 1) // 2

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    moleham = of.InteractionOperator(0, soei, 0.25 * astei)
    redham = of.make_reduced_hamiltonian(moleham, n_electrons=4)

    topdm = of.map_two_pdm_to_one_pdm(tpdm, 4)
    assert np.allclose(topdm, opdm)
    assert np.allclose(sopdm, opdm[::2, ::2] + opdm[1::2, 1::2])

    stepper = Stepper(method)

    if verbose:
        print("Initial Energy")
        print(np.einsum('ijkl,ijkl', 0.5 * tei, stpdm) + np.einsum('ij,ij', oei, sopdm))
        print("Entering Newton Steps:")

    orb_rotations = []
    current_iter = 0
    while grad_residual > grad_eps and current_iter < maxiter:
        comm_grad = np.zeros(num_vars)

        k2_tensor = redham.two_body_tensor.real.transpose(0, 1, 3, 2)
        v2_tensor = moleham.two_body_tensor.transpose(0, 1, 3, 2)
        h1_tensor = moleham.one_body_tensor

        for idx, (v, o) in enumerate(zip(ridx, cidx)):
            comm_grad_val = spinless_rotgrad_onebody(oei, v, o, sopdm)
            comm_grad_val += 0.5 * spinless_rotgrad_twobody(tei, v, o, stpdm)
            comm_grad[idx] = comm_grad_val

            test_comm_grad_val = k2_rotgen_grad(k2_tensor, 2 * v, 2 * o, tpdm) + k2_rotgen_grad(k2_tensor, 2 * v + 1, 2 * o + 1, tpdm)
            assert np.isclose(test_comm_grad_val, comm_grad_val)

        grad_residual = np.linalg.norm(comm_grad)

        comm_hess = np.zeros((num_vars, num_vars))
        for idx, (v1, o1) in enumerate(zip(ridx, cidx)):
            for jdx, (v2, o2) in enumerate(zip(ridx, cidx)):
                if idx >= jdx:
                    hess_val = 0.
                    hess_val += 0.5 * spinless_rothess_twobody(tei, v1, o1, v2, o2,
                                                               stpdm)
                    hess_val += 0.5 * spinless_rothess_twobody(tei, v2, o2, v1, o1,
                                                               stpdm)
                    hess_val += spinless_rothess_onebody(oei, v1, o1, v2, o2, sopdm)
                    hess_val += spinless_rothess_onebody(oei, v2, o2, v1, o1, sopdm)

                    comm_hess[idx, jdx] = 0.5 * hess_val
                    comm_hess[jdx, idx] = 0.5 * hess_val

                    true_hess_val_ob = 0.
                    true_hess_val_tb = 0.

                    assert np.allclose(sopdm, opdm[::2, ::2] + opdm[1::2, 1::2])
                    for sigma, tau in product(range(2), repeat=2):
                        true_hess_val_tb += k2_rotgen_hessian(
                            v2_tensor, 2 * v1 + sigma, 2 * o1 + sigma,
                                       2 * v2 + tau, 2 * o2 + tau,
                            tpdm
                        )
                        true_hess_val_tb += k2_rotgen_hessian(
                            v2_tensor, 2 * v2 + sigma, 2 * o2 + sigma,
                                       2 * v1 + tau, 2 * o1 + tau,
                            tpdm
                        )
                        true_hess_val_ob += k2_rotgen_hess_one_body(h1_tensor,
                                                            2 * v1 + sigma,
                                                            2 * o1 + sigma,
                                                            2 * v2 + tau,
                                                            2 * o2 + tau, opdm)
                        true_hess_val_ob += k2_rotgen_hess_one_body(h1_tensor,
                                                            2 * v2 + sigma,
                                                            2 * o2 + sigma,
                                                            2 * v1 + tau,
                                                            2 * o1 + tau, opdm)
                    tval_ob = spinless_rothess_onebody(oei, v1, o1, v2, o2, sopdm) + spinless_rothess_onebody(oei, v2, o2, v1, o1, sopdm)
                    tval_tb = 0.5 * (spinless_rothess_twobody(tei, v1, o1, v2, o2, stpdm) + spinless_rothess_twobody(tei, v2, o2, v1, o1, stpdm))
                    # print(true_hess_val_ob, tval_ob)
                    assert np.isclose(true_hess_val_ob, tval_ob)
                    # print(true_hess_val_tb, tval_tb)
                    assert np.isclose(true_hess_val_tb, tval_tb)


        assert np.allclose(comm_hess, comm_hess.T)

        new_fr_vals = stepper(comm_grad, comm_hess)

        new_kappa = np.zeros((norbs, norbs))
        new_kappa[ridx, cidx] = new_fr_vals
        new_kappa -= new_kappa.T  # upper triangle has negative signs

        basis_change = expm(new_kappa).real
        oei = of.general_basis_change(oei, basis_change, (1, 0))
        tei = of.general_basis_change(tei, basis_change, (1, 1, 0, 0))
        redham.rotate_basis(basis_change)
        moleham.rotate_basis(basis_change)

        orb_rotations.append(basis_change)
        current_iter += 1

        if verbose:
            print("\t{}\t{:5.10f}\t{:5.10f}".format(
                current_iter,
                np.einsum('ijkl,ijkl', 0.5 * tei, stpdm) + np.einsum('ij,ij', oei, sopdm),
                grad_residual)
            )


    # no list reversal required because
    # e^{-KN}...e^{-K2}e^{-K1}He^{K1}e^{K2}e^{K3}...e^{KN}
    # thus total rotation is e^{K1}e^{K2}...e^{KN} which we can implement we
    # just normal reduce
    return reduce(np.dot, orb_rotations)


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


def main():
    np.random.seed(0)
    molecule = load_hydrogen_fluoride_molecule(1.7)
    norbs = molecule.n_orbitals
    hf_opdm = np.diag([1] * molecule.n_electrons + [0] * (2 * molecule.n_orbitals - molecule.n_electrons))
    hf_tpdm = 2 * of.wedge(hf_opdm, hf_opdm, (1, 1), (1, 1)).real
    nelec = molecule.n_electrons
    sz = 0

    fqe_wf = fqe.Wavefunction([[nelec, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.sector((nelec, sz)).coeff = fqe_wf.sector((nelec, sz)).coeff.real
    fqe_wf.normalize()
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).real
    opdm, tpdm = fqe_wf.sector((nelec, sz)).get_openfermion_rdms()
    opdm = opdm.real
    tpdm = tpdm.real

    pyscf_scf =  molecule._pyscf_data['scf']
    pyscf_molecule = molecule._pyscf_data['mol']
    S = pyscf_scf.get_ovlp()
    Hcore = pyscf_scf.get_hcore()
    # Rotate back to AO basis
    ao_eri = pyscf_molecule.intor('int2e', aosym='s1')  # (ij|kl)
    _, X = sp.linalg.eigh(Hcore, S)
    obi = of.general_basis_change(Hcore, X, (1, 0))
    tbi = np.einsum('psqr', of.general_basis_change(ao_eri, X, (1, 0, 1, 0)))
    fqe_ham = RestrictedHamiltonian((obi, np.einsum("ijlk", -0.5 * tbi)))
    soei, stei = spinorb_from_spatial(obi, tbi)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    moleham = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.make_reduced_hamiltonian(moleham,
                                              molecule.n_electrons)

    sopdm = opdm[::2, ::2] + opdm[1::2, 1::2]
    stpdm = tpdm[::2, ::2, ::2, ::2] + tpdm[1::2, 1::2, 1::2, 1::2] + \
             tpdm[::2, 1::2, 1::2, ::2] + tpdm[1::2, ::2, ::2, 1::2]

    # sopdm = hf_opdm[::2, ::2] + hf_opdm[1::2, 1::2]
    # stpdm = hf_tpdm[::2, ::2, ::2, ::2] + hf_tpdm[1::2, 1::2, 1::2, 1::2] + \
    #         hf_tpdm[::2, 1::2, 1::2, ::2] + hf_tpdm[1::2, ::2, ::2, 1::2]


    fqe_hf_wf = fqe.Wavefunction([[nelec, sz, norbs]])
    fqe_hf_wf.set_wfn(strategy='hartree-fock')

    # unitary = orbital_optimize_spinorb(reduced_ham=reduced_ham, tpdm=hf_tpdm,
    #                                    verbose=True, method='newton-raphson')
    unitary = oo_so_sep(obi.copy(), tbi.copy(), tpdm.copy(), verbose=True, method='augmented-hessian')
    unitary = spatial_oo(obi.copy(), tbi.copy(), sopdm, stpdm, tpdm=tpdm.copy(), opdm=opdm.copy(), verbose=True, method='augmented-hessian')
    print(molecule.hf_energy - molecule.nuclear_repulsion)

if __name__ == "__main__":
    main()