import fqe
import numpy as np
import scipy as sp
from scipy.sparse.coo import coo_matrix

import openfermion as of
# from openfermion.hamiltonians.richardson_gaudin import RichardsonGaudin
from openfermion.chem.molecular_data import spinorb_from_spatial

from fqe.openfermion_utils import integrals_to_fqe_restricted

from itertools import chain, product

# from qcpanop.cc.lambda_ccd import (kernel, ccsd_d1, ccsd_d2, ccsd_energy,
#                                    lagrangian_energy)
from lambda_ccd import (kernel, ccsd_d1, ccsd_d2, ccsd_energy,
                                   lagrangian_energy)
from lambda_ccd import singles_residual
from lambda_ccd import doubles_residual

from lambda_ccd_2 import kernel as kernel2
from lambda_ccd_2 import ccsd_energy as ccsd_energy2
from lambda_ccd_2 import lagrangian_energy as lagrangian_energy2
from lambda_ccd_2 import singles_residual as singles_residual2
from lambda_ccd_2 import doubles_residual as doubles_residual2

import matplotlib.pyplot as plt

splus = lambda xx: of.QubitOperator((xx, 'X'), coefficient=0.5) + of.QubitOperator((xx, 'Y'), coefficient=-0.5j)
sminus = lambda xx: of.QubitOperator((xx, 'X'), coefficient=0.5) + of.QubitOperator((xx, 'Y'), coefficient=0.5j)
sz = lambda xx: of.QubitOperator((xx, 'Z'), coefficient=1)


def drop_identity(qubit_ham):
    new_ham = of.QubitOperator()
    for term in qubit_ham:
        if not of.is_identity(term):
            new_ham += term
    return new_ham


def get_num_projector(n_qubits, n_target):
    n_row_idx = []
    for ii in range(2**n_qubits):
        ket = [int(xx) for xx in np.binary_repr(ii, width=n_qubits)]
        if sum(ket) == n_target:
            n_row_idx.append(ii)
    n_projector = coo_matrix(([1] * len(n_row_idx), (n_row_idx, n_row_idx)),
                             shape=(2 ** n_qubits, 2 ** n_qubits))
    return n_projector

def get_sz_projector(n_qubits, n_target):
    n_row_idx = []
    for ii in range(2**n_qubits):
        ket = np.array([int(xx) for xx in np.binary_repr(ii, width=n_qubits)])
        keta = ket[::2]
        ketb = ket[1::2]
        if np.isclose(sum(keta) - sum(ketb), n_target):
            n_row_idx.append(ii)
    n_projector = coo_matrix(([1] * len(n_row_idx), (n_row_idx, n_row_idx)),
                             shape=(2 ** n_qubits, 2 ** n_qubits))
    return n_projector

def get_doci_projector(n_qubits, n_target):
    """Get the projector on to the doubly occupied space

    :param int n_qubits: number of qubits (or fermionic modes)
    :param int n_target: number of electrons total
    :return: coo_matrix that is 1 along diagonal elements that correspond
             to doci terms.
    """
    n_row_idx = []
    for ii in range(2**n_qubits):
        ket = np.binary_repr(ii, width=n_qubits)
        keta = ket[::2]
        ketb = ket[1::2]
        res = int(keta, 2) ^ int(ketb, 2)
        if np.isclose(res, 0) and ket.count('1') == n_target:
            n_row_idx.append(ii)
    n_projector = coo_matrix(([1] * len(n_row_idx), (n_row_idx, n_row_idx)),
                             shape=(2 ** n_qubits, 2 ** n_qubits))
    return n_projector, n_row_idx


def get_gs(ham, projectors=None, sector_op=None, sector_n=None):
    dense_ham = of.get_sparse_operator(ham).real
    for proj in projectors:
        dense_ham = proj.T @ dense_ham @ proj


    w, v = np.linalg.eigh(dense_ham.toarray())

    if sector_op is None and sector_n is None:
        return w[0], v[:, [0]]
    else:
        for ii in range(len(w)):
            n_val = v[:, [ii]].conj().T @ sector_op @ v[:, [ii]]
            # print("Sector selector ", n_val)
            if np.isclose(n_val, sector_n):
                return w[ii], v[:, [ii]]
        else:
            raise ValueError("Didn't find desired sector")


def print_wf(wf):
    n_qubits = int(np.log2(wf.shape[0]))
    for ii in range(2**n_qubits):
        if not np.isclose(np.abs(wf[ii]), 0):
            print(ii, np.binary_repr(ii, width=n_qubits), wf[ii])


def get_fermion_operator(rg_ham):
    oei, tei = rg_ham.get_projected_integrals()
    ele_ham = integrals_to_fqe_restricted(oei, tei)
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    io_ham = of.InteractionOperator(0, soei, 0.25 * astei)
    return io_ham, ele_ham


def qubit_wf_to_fqe(wf):
    """Convert to full space wf then to fqe"""
    n_qubits = int(np.log2(wf.shape[0]))
    full_space_wf = np.zeros((2**(2 * n_qubits)))
    for ii in range(2**n_qubits):
        doci_space_ket = [int(xx) for xx in np.binary_repr(ii, width=n_qubits)]
        full_space_ket = list(chain(*zip(doci_space_ket, doci_space_ket)))
        full_space_idx = int("".join([str(xx) for xx in full_space_ket]), 2)
        full_space_wf[full_space_idx] = wf[ii]
    # fqe_wf = fqe.from_cirq(full_space_wf, 1.0E-12)
    fqe_wf = None
    return fqe_wf, full_space_wf


def get_rg_fham(g, spatial_orbs):
    h1 = np.diag(np.arange(spatial_orbs) + 1)
    h1 = np.kron(h1, np.eye(2))
    h2 = np.zeros((2 * spatial_orbs,) * 4)
    fham = of.FermionOperator()
    for pp in range(spatial_orbs):
        fham += of.FermionOperator(((2 * pp, 1), (2 * pp, 0)), coefficient=float(h1[2 * pp, 2 * pp]))
        fham += of.FermionOperator(((2 * pp + 1, 1), (2 * pp + 1, 0)), coefficient=float(h1[2 * pp + 1, 2 * pp + 1]))
    for p, q in product(range(spatial_orbs), repeat=2):
        if p != q:
            h2[2 * p, 2 * p + 1, 2 * q + 1, 2 * q] = g / 2
            h2[2 * p + 1, 2 * p, 2 * q, 2 * q + 1] = g / 2
            ab = ((2 * p, 1), (2 * p + 1, 1), (2 * q + 1, 0), (2 * q, 0))
            ba = ((2 * p + 1, 1), (2 * p, 1), (2 * q, 0), (2 * q + 1, 0))
            fham += of.FermionOperator(ab, coefficient=g/2)
            fham += of.FermionOperator(ba, coefficient=g/2)
    return h1, h2, fham


def get_rg_qham(g, spatial_orbs):
    """
    sum_{p}e_{p}~N_{p} + g sum_{pq} P_{p}^P_{q}
    :param g:
    :param spatial_orbs:
    :return:
    """
    qham = of.QubitOperator()
    h1 = np.diag(np.arange(spatial_orbs) + 1 )
    # constant term
    constant = sum(np.diagonal(h1))
    for pp in range(spatial_orbs):
        qham += of.QubitOperator((pp, 'Z'), coefficient=float(-h1[pp, pp]))
        for qq in range(spatial_orbs):
            if pp != qq:
                qham += of.QubitOperator(((pp, 'X'), (qq, 'X')), coefficient=g/4)
                qham += of.QubitOperator(((pp, 'Y'), (qq, 'Y')), coefficient=g/4)
    return qham, float(constant)


def solve_cc_equations2(soei, astei):
    """Lambda-CCD equations are solved. Return amplitudes and RDMs"""
    nso = soei.shape[0]
    nsocc = nso // 2
    nsvirt = nso - nsocc
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    fock = soei + np.einsum('piiq->pq', 4 * astei[:, o, o, :])  # the CC equations generated in ccd_2 don't involve the 1/4 term
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])

    eps = np.diagonal(fock)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])
    g = astei.transpose(0, 1, 3, 2)
    t1z, t2z = np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1f, t2f, l1f, l2f = kernel2(t1z, t2z, soei, g, o, v, e_ai, e_abij,
                                stopping_eps=1.0E-6, max_iter=300, damping=0.5)
    assert np.isclose(np.linalg.norm(t1f), 0)
    assert np.isclose(np.linalg.norm(l1f), 0)
    print("{: 5.15f} HF Energy ".format(hf_energy))
    final_cc_energy = ccsd_energy2(t1f, t2f, soei, g, o, v) - hf_energy
    print("{: 5.15f} Final Correlation Energy".format(final_cc_energy))
    final_lagrangian_energy = lagrangian_energy2(t1f, t2f, l1f, l2f, soei, g, o, v) - hf_energy
    print("{: 5.15f} Lagrangian Energy - HF".format(final_lagrangian_energy))

    kd = np.eye(nso)
    opdm = ccsd_d1(t1f, t2f, l1f, l2f, kd, o, v)
    tpdm = ccsd_d2(t1f, t2f, l1f, l2f, kd, o, v)
    tpdm = tpdm.transpose(0, 1, 3, 2) # openfermion ordering

    return final_cc_energy + hf_energy, final_lagrangian_energy + hf_energy, opdm, tpdm, t2f, l2f


def solve_cc_equations(soei, astei):
    """Lambda-CCD equations are solved. Return amplitudes and RDMs"""
    nso = soei.shape[0]
    nsocc = nso // 2
    nsvirt = nso - nsocc
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    eps = np.diag(fock)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])
    g = astei.transpose(0, 1, 3, 2)
    t1z, t2z = np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1f, t2f, l1f, l2f = kernel(t1z, t2z, fock, g, o, v, e_ai, e_abij,
                                stopping_eps=1.0E-6, damping=0.5, max_iter=300)
    assert np.isclose(np.linalg.norm(t1f), 0)
    assert np.isclose(np.linalg.norm(l1f), 0)
    print("{: 5.15f} HF Energy ".format(hf_energy))
    final_cc_energy = ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy
    print("{: 5.15f} Final Correlation Energy".format(final_cc_energy))
    final_lagrangian_energy = lagrangian_energy(t1f, t2f, l1f, l2f, fock, g, o, v) - hf_energy
    print("{: 5.15f} Lagrangian Energy - HF".format(final_lagrangian_energy))

    kd = np.eye(nso)
    opdm = ccsd_d1(t1f, t2f, l1f, l2f, kd, o, v)
    tpdm = ccsd_d2(t1f, t2f, l1f, l2f, kd, o, v)
    tpdm = tpdm.transpose(0, 1, 3, 2) # openfermion ordering

    return final_cc_energy + hf_energy, final_lagrangian_energy + hf_energy, opdm, tpdm, t1f, t2f, l1f, l2f


def doci_vec_to_full_space_pyscf(doci_vec, doci_index, n_qubits):
    """This function is very slow because of from_cirq.  This will
    be sped up in the future."""
    wf = np.zeros((2**n_qubits), dtype=np.complex128)
    for idx, val in enumerate(doci_index):
        wf[val] = doci_vec[idx]

    fqe_wf = fqe.from_cirq(wf, thresh=1.0E-12)
    return fqe_wf


def main():
    # set up couplings and simulation parameters
    couplings = np.linspace(-0.5, 1.5, 20)
    n_qubits = 6# this is for the DOCI space so equivalent to spatial orbs
    nmo = n_qubits

    # results storage
    ccd_energies = []
    ccd_energies2 = []
    doci_energies = []
    true_sc_order_parameter = []
    ccd_sc_order_parameter = []

    doci_projector, s0_basis = get_doci_projector(2 * n_qubits, n_qubits)
    # run through everything
    for g in couplings:
        print("Coupling parameter ", g)
        ncr_h1, ncr_h2, ncr_fham = get_rg_fham(g, n_qubits)

        # construct antisymmetric coefficient matrix
        # such that the operator commutes with antisymmeterizer
        ncr_h2_antisymm = ncr_h2 - np.einsum('ijlk', ncr_h2)
        for p, q, r, s in product(range(2 * n_qubits), repeat=4):
            if not np.isclose(ncr_h2_antisymm[p, q, r, s], 0):
                # print((p, q, r, s), ncr_h2_antisymm[p, q, r, s])
                assert np.isclose(ncr_h2_antisymm[p, q, r, s], -ncr_h2_antisymm[p, q, s, r])
                assert np.isclose(ncr_h2_antisymm[p, q, r, s], -ncr_h2_antisymm[q, p, r, s])
                assert np.isclose(ncr_h2_antisymm[p, q, r, s],  ncr_h2_antisymm[q, p, s, r])

        # get interaction operator
        # normally a factor of 1/4 is associated with antisymmetric 2-electron
        # integrals.  Here is is 1/2 (because we only consider alpha-beta space)
        ncr_antisymm_fham = of.InteractionOperator(0, ncr_h1.astype(float),
                                                   0.5 * ncr_h2_antisymm.astype(
                                                       float))
        test1 = of.normal_ordered(of.get_fermion_operator(ncr_antisymm_fham))
        test2 = of.normal_ordered(ncr_fham)
        assert test1 == test2  # check if equivalent to the original HamiltonianA

        # diagonalize in DOCI space.
        afham = of.get_sparse_operator(ncr_antisymm_fham).toarray().real
        afham = afham[:, doci_projector.row]
        afham = afham[doci_projector.row, :]
        ncr_afham_eigs, ncr_afham_vecs = np.linalg.eigh(afham)
        doci_energies.append(ncr_afham_eigs[0])

        full_space_wf = np.zeros((4 ** nmo), dtype=np.complex128)
        for idx in range(len(s0_basis)):
            full_space_wf[s0_basis[idx]] = ncr_afham_vecs[idx, 0]
        # print wf
        fqe_doci = fqe.from_cirq(full_space_wf.flatten(), thresh=1.0E-12)
        fqe_doci.print_wfn()
        # check S0 wf gives the same expectation value
        fqe_fham = fqe.get_hamiltonian_from_openfermion(ncr_fham)
        doci_energy = fqe_doci.expectationValue(fqe_fham)
        assert np.isclose(doci_energy, ncr_afham_eigs[0])

        # get FCI RDMs
        fqe_wf = doci_vec_to_full_space_pyscf(ncr_afham_vecs[:, [0]].flatten(),
                                              doci_projector.row, 2 * n_qubits)
        fqe_wf.print_wfn()

        fqe_opdm, fqe_tpdm = fqe_wf.sector((n_qubits, 0)).get_openfermion_rdms()
        true_num_fluctation = (2 / n_qubits) * np.sum(np.sqrt(np.diagonal(fqe_opdm) - np.diagonal(fqe_opdm)**2))
        true_sc_order_parameter.append(true_num_fluctation)


        # print()
        # solve CCD equations
        cc_energy, lagrangian_energy, cc_opdm, cc_tpdm, cc_t1f, cc_t2f, cc_l1f, cc_l2f = solve_cc_equations(
            ncr_h1.astype(float), 4 * 0.5 * ncr_h2_antisymm.astype(float))
        ccd_energies.append(cc_energy)
        ccd_num_fluctation = (2 / n_qubits) * np.sum(np.sqrt(np.diagonal(cc_opdm) - np.diagonal(cc_opdm)**2))
        ccd_sc_order_parameter.append(ccd_num_fluctation)

        # Solving ccsd_v2
        nso = ncr_h1.shape[0]
        nsocc = nso // 2
        nsvirt = nso - nsocc
        n = np.newaxis
        o = slice(None, nsocc)
        v = slice(nsocc, None)
        fock = ncr_h1 + np.einsum('piiq->pq', 2 * ncr_h2_antisymm[:, o, o, :])
        print("CC-S",
              np.linalg.norm(singles_residual(cc_t1f, cc_t2f, fock, 2 * ncr_h2_antisymm.transpose(0, 1, 3, 2), o, v)))
        print("CC-D",
              np.linalg.norm(doubles_residual(cc_t1f, cc_t2f, fock, 2 * ncr_h2_antisymm.transpose(0, 1, 3, 2), o, v)))

        print("CC-S",
              np.linalg.norm(singles_residual2(cc_t1f, cc_t2f, ncr_h1, 0.5 * ncr_h2_antisymm.transpose(0, 1, 3, 2), o, v)))
        print("CC-D",
              np.linalg.norm(doubles_residual2(cc_t1f, cc_t2f, ncr_h1, 0.5 * ncr_h2_antisymm.transpose(0, 1, 3, 2), o, v)))

        cc_energy, lagrangian_energy, cc_opdm, cc_tpdm, cc_t2f, cc_l2f = solve_cc_equations2(
            ncr_h1.astype(float), 0.5 * ncr_h2_antisymm.astype(float))
        ccd_energies2.append(cc_energy)



    np.save("RG_n_{}_gvalues.npy".format(n_qubits), couplings)
    np.save("RG_n_{}_doci_energy.npy".format(n_qubits), np.array(doci_energies))
    np.save("RG_n_{}_ccd_energies.npy".format(n_qubits), np.array(ccd_energies))
    np.save("RG_n_{}_ccd_energies2.npy".format(n_qubits), np.array(ccd_energies2))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8))
    ax.plot(couplings, doci_energies, 'k', linestyle='-', label='DOCI')
    ax.plot(couplings, ccd_energies,  marker='s', linestyle='None', label='CCSD-f-v', mfc='C0', mec='None')
    ax.plot(couplings, ccd_energies2, marker='o', linestyle='None', label='CCSD-h-g', mfc='C1', mec='None')

    ax.tick_params(which='both', labelsize=18, direction='in')
    ax.set_xlabel("g", fontsize=18)
    ax.set_ylabel(r"E", fontsize=18)
    ax.legend(loc='center', fontsize=13, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("RG_n_{}_absolute_ccd_doci_energies.png".format(n_qubits), format='PNG', dpi=300)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8))
    ax.plot(couplings, np.array(ccd_energies).flatten() - doci_energies, 'C0o-', label='CCSD-f-v')
    ax.plot(couplings, np.array(ccd_energies2).flatten() - doci_energies, 'C1o-', label='CCSD-h-g')
    ax.tick_params(which='both', labelsize=18, direction='in')
    ax.set_xlabel("g", fontsize=18)
    ax.set_ylabel(r"$E - E_{\mathrm{exact}}$", fontsize=18)
    plt.axhline(0, color='k')
    ax.legend(loc='upper right', fontsize=13, ncol=2)
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.tight_layout()
    plt.savefig("RG_n_{}_relative_ccd_doci_energies.png".format(n_qubits), format='PNG', dpi=300)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8))
    ax.plot(couplings, ccd_sc_order_parameter, 'C0o-', label='CCD')
    ax.plot(couplings, true_sc_order_parameter, 'k-', label='FCI')
    ax.tick_params(which='both', labelsize=18, direction='in')
    ax.set_xlabel("g", fontsize=18)
    ax.set_ylabel(r"$\Delta_{b}$", fontsize=18)
    plt.axhline(0, color='k')
    ax.legend(loc='upper right', fontsize=13, ncol=2)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("RG_n_{}_superconducting_correlation_l4.png".format(n_qubits), format='PNG', dpi=300)
    plt.show()




if __name__ == "__main__":
    main()