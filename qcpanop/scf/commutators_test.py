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

from qcpanop.scf.commutators import (k2_rotgen_grad,
                                     k2_rotgen_grad_one_body,
                                     k2_rotgen_hessian,
                                     k2_rotgen_hess_one_body,
                                     spinless_rotgrad_onebody,
                                     spinless_rotgrad_twobody,
                                     spinless_rothess_onebody,
                                     spinless_rothess_twobody,
                                     )


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


def test_check_antisym(molham):
    nsorbs = molham.one_body_tensor.shape[0]
    h1 = molham.one_body_tensor
    v2 = molham.two_body_tensor

    for p, q, r, s in product(range(nsorbs), repeat=4):
        assert np.isclose(v2[p, q, r, s], -v2[q, p, r, s])
        assert np.isclose(v2[p, q, r, s], -v2[p, q, s, r])


def test_hamm_one_body_commutator(redham, moleham):
    k2mat = of.get_sparse_operator(redham).real
    k2_tensor = redham.two_body_tensor.transpose(0, 1, 3, 2)
    v2_tensor = moleham.two_body_tensor.transpose(0, 1, 3, 2)
    h1_tensor = moleham.one_body_tensor
    nsorbs = k2_tensor.shape[0]
    norbs = k2_tensor.shape[0] // 2
    nelec = 4
    sz = 0


    v2_moleham = copy.deepcopy(moleham)
    h1_moleham = copy.deepcopy(moleham)
    v2_moleham.one_body_tensor = np.zeros_like(v2_moleham.one_body_tensor)
    h1_moleham.two_body_tensor = np.zeros_like(h1_moleham.two_body_tensor)
    v2mat = of.get_sparse_operator(v2_moleham, n_qubits=nsorbs)
    h1mat = of.get_sparse_operator(h1_moleham, n_qubits=nsorbs)

    assert np.allclose(k2_tensor.transpose(0, 1, 3, 2), redham.two_body_tensor)

    fqe_wf = fqe.Wavefunction([[nelec, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.sector((nelec, sz)).coeff = fqe_wf.sector((nelec, sz)).coeff.real
    fqe_wf.normalize()
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).real
    opdm, tpdm = fqe_wf.sector((nelec, sz)).get_openfermion_rdms()
    assert np.isclose(cirq_wf.T @ k2mat @ cirq_wf,
                      np.einsum('ijlk,ijkl', k2_tensor, tpdm))

    up_op = [of.FermionOperator(((xx, 1))) for xx in range(2 * norbs)]
    up_op = [of.get_sparse_operator(op, n_qubits=2 * norbs) for op in up_op]
    dwn_op = [op.conj().T for op in up_op]

    # This loop will take a long time but we can check everything
    for p, q in product(range(nsorbs), repeat=2):
        test_val = k2_rotgen_grad(k2_tensor, p, q, tpdm)
        fop = up_op[p] @ dwn_op[q] - up_op[q] @ dwn_op[p]
        true_val = cirq_wf.T @ of.commutator(k2mat, fop) @ cirq_wf
        assert np.isclose(test_val, true_val)

        test_val = k2_rotgen_grad(v2_tensor, p, q, tpdm)
        true_val = cirq_wf.T @ of.commutator(v2mat, fop) @ cirq_wf
        assert np.isclose(test_val, true_val)

        test_val = k2_rotgen_grad_one_body(h1_tensor, p, q, opdm)
        true_val = cirq_wf.T @ of.commutator(h1mat, fop) @ cirq_wf
        assert np.isclose(test_val, true_val)

        for r, s in product(range(nsorbs), repeat=2):
            print(p, q, r, s)
            test_val = k2_rotgen_hessian(k2_tensor, p, q, r, s, tpdm)
            fop_outer = up_op[r] @ dwn_op[s] - up_op[s] @ dwn_op[r]
            true_val = cirq_wf.T @ of.commutator(of.commutator(k2mat, fop), fop_outer) @ cirq_wf
            assert np.isclose(test_val, true_val)

            test_val = k2_rotgen_hessian(v2_tensor, p, q, r, s, tpdm)
            fop_outer = up_op[r] @ dwn_op[s] - up_op[s] @ dwn_op[r]
            true_val = cirq_wf.T @ of.commutator(of.commutator(v2mat, fop), fop_outer) @ cirq_wf
            assert np.isclose(test_val, true_val)

            test_val = k2_rotgen_hess_one_body(h1_tensor, p, q, r, s, opdm)
            fop_outer = up_op[r] @ dwn_op[s] - up_op[s] @ dwn_op[r]
            true_val = cirq_wf.T @ of.commutator(of.commutator(h1mat, fop), fop_outer) @ cirq_wf
            assert np.isclose(test_val, true_val)


def test_spin_summed_commutators(oei, tei):
    nelec = 4
    sz = 0
    norbs = oei.shape[0]
    fqe_wf = fqe.Wavefunction([[nelec, sz, norbs]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.sector((nelec, sz)).coeff = fqe_wf.sector((nelec, sz)).coeff.real
    fqe_wf.normalize()
    cirq_wf = fqe.to_cirq_ncr(fqe_wf).real
    opdm, tpdm = fqe_wf.sector((nelec, sz)).get_openfermion_rdms()
    opdm = opdm.real
    tpdm = tpdm.real

    sopdm = opdm[::2, ::2] + opdm[1::2, 1::2]
    stpdm = tpdm[::2, ::2, ::2, ::2] + tpdm[1::2, 1::2, 1::2, 1::2] + \
            tpdm[::2, 1::2, 1::2, ::2] + tpdm[1::2, ::2, ::2, 1::2]


    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    moleham = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.make_reduced_hamiltonian(moleham,
                                              molecule.n_electrons)
    v2_moleham = copy.deepcopy(moleham)
    h1_moleham = copy.deepcopy(moleham)
    v2_moleham.one_body_tensor = np.zeros_like(v2_moleham.one_body_tensor)
    h1_moleham.two_body_tensor = np.zeros_like(h1_moleham.two_body_tensor)
    v2mat = of.get_sparse_operator(v2_moleham, n_qubits=2 * norbs)
    h1mat = of.get_sparse_operator(h1_moleham, n_qubits=2 * norbs)

    print((cirq_wf.T @ (h1mat) @ cirq_wf).real)
    print(np.einsum('ij,ij', oei, sopdm))

    print((cirq_wf.T @ v2mat @ cirq_wf).real)
    print(np.einsum('ijkl,ijkl', 0.5 * tei, stpdm))

    print((cirq_wf.T @ (h1mat + v2mat) @ cirq_wf).real)
    print(np.einsum('ij,ij', oei, sopdm) + np.einsum('ijkl,ijkl', 0.5 * tei, stpdm))

    for p, q in product(range(norbs), repeat=2):
        test_val = spinless_rotgrad_onebody(oei, p, q, sopdm)
        true_val = k2_rotgen_grad_one_body(moleham.one_body_tensor, 2 * p, 2 * q, opdm) + \
                   k2_rotgen_grad_one_body(moleham.one_body_tensor, 2 * p + 1,
                                           2 * q + 1, opdm)
        # print(test_val, true_val)
        assert np.isclose(test_val, true_val)

        test_val = 0.5 * spinless_rotgrad_twobody(tei, p, q, stpdm)
        true_val = k2_rotgen_grad(moleham.two_body_tensor.transpose(0, 1, 3, 2), 2 * p, 2 * q, tpdm) + \
                   k2_rotgen_grad(moleham.two_body_tensor.transpose(0, 1, 3, 2), 2 * p + 1,
                                           2 * q + 1, tpdm)
        assert np.isclose(test_val, true_val)

        for r, s in product(range(norbs), repeat=2):
            test_val = spinless_rothess_onebody(oei, p, q, r, s, sopdm)
            true_val = 0
            true_val += k2_rotgen_hess_one_body(moleham.one_body_tensor, 2 * p + 0, 2 * q + 0, 2 * r + 0, 2 * s + 0, opdm)
            true_val += k2_rotgen_hess_one_body(moleham.one_body_tensor, 2 * p + 0, 2 * q + 0, 2 * r + 1, 2 * s + 1, opdm)
            true_val += k2_rotgen_hess_one_body(moleham.one_body_tensor, 2 * p + 1, 2 * q + 1, 2 * r + 0, 2 * s + 0, opdm)
            true_val += k2_rotgen_hess_one_body(moleham.one_body_tensor, 2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1, opdm)
            assert np.isclose(test_val, true_val)

            test_val = 0.5 * spinless_rothess_twobody(tei, p, q, r, s, stpdm)
            true_val = 0
            for sigma, tau in product(range(2), repeat=2):
                true_val += k2_rotgen_hessian(
                    moleham.two_body_tensor.transpose(0, 1, 3, 2),
                    2 * p + sigma, 2 * q + sigma,
                    2 * r + tau, 2 * s + tau,
                    tpdm
                )
            # print(test_val, true_val)
            assert np.isclose(test_val, true_val)






if __name__ == "__main__":
    molecule = load_hydrogen_fluoride_molecule(1.7)
    norbs = molecule.n_orbitals
    hf_opdm = np.diag([1] * molecule.n_electrons + [0] * (2 * molecule.n_orbitals - molecule.n_electrons))
    hf_tpdm = 2 * of.wedge(hf_opdm, hf_opdm, (1, 1), (1, 1)).real

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
    test_check_antisym(moleham)
    # test_hamm_one_body_commutator(reduced_ham, moleham)
    test_spin_summed_commutators(obi, tbi)
    exit()
