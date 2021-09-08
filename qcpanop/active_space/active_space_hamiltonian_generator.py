"""
Select Active space and save to disk

We don't need to use localized orbitals for this. We should check if it is different
"""
import h5py
import copy
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.mcscf import avas
from pyscf.fci.cistring import make_strings

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted


def get_rohf_active_space_hamiltonian_pyscf(mf, ncas, nelecas, avas_orbs=None):
    assert isinstance(mf, scf.rohf.ROHF)
    cas = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    h1, ecore = cas.get_h1eff(mo_coeff=avas_orbs)
    eri = cas.get_h2cas(mo_coeff=avas_orbs)
    eri = ao2mo.restore('s1', eri, h1.shape[0])  # chemist convention (11|22)
    ecore = np.float(ecore)
    return ecore, h1, eri


def active_space_ham_from_pyscf_ints(h1, eri, ecore=0.):
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    tei = np.asarray(
        eri.transpose(0, 2, 3, 1), order='C')
    soei, stei = spinorb_from_spatial(h1, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    active_space_ham = of.InteractionOperator(ecore, soei, 0.25 * astei)
    return active_space_ham


def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = [x for x in make_strings(norb_list, nelec[0])]
    n_beta_strings = [x for x in make_strings(norb_list, nelec[1])]

    fqe_wf_ci = fqe.Wavefunction([[sum(nelec), nelec[0] - nelec[1], norbs]])
    fqe_data_ci = fqe_wf_ci.sector((sum(nelec), nelec[0] - nelec[1]))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()), dtype=np.complex128)
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            # if np.abs(civec[paidx, pbidx]) > 1.0E-3:
            #     print(np.binary_repr(pyscf_alpha_idx, width=10), np.binary_repr(pyscf_beta_idx, width=10), civec[paidx, pbidx])
            fqe_orderd_coeff[fqe_graph_ci.index_alpha(
                pyscf_alpha_idx), fqe_graph_ci.index_beta(pyscf_beta_idx)] = \
                pyscf_cimat[paidx, pbidx]

    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci

def active_space_generator(chkfile_path, sys_name, scf_type, spin, basis, loc_type, occ_orbitals, virtual_orbitals):
    """
    Generate active space form pyscf checkpoint file

    :param chkfile_path: path to checkpoitnn file
    :param sys_name:  name of the system
    :param scf_type: scf type of the system
    :param spin:  spin value (2S) like in pyscf
    :param basis: basis that was used
    :param loc_type: if localization is used specify.
    :param occ_orbitals: occupied orbitals
    :param virtual_orbitals: virtual orbitals

    Writes a h5py file that stores the active space Hamiltonian
    """
    mol, scf_dict = scf.chkfile.load_scf(chkfile_path)

    active_norb = len(occ_orbitals) + len(virt_orbitals)
    active_ne = sum([scf_dict['mo_occ'][xx] for xx in occ_orbitals])
    total_electrons = int(sum(scf_dict['mo_occ']))
    nalpha_total = len(np.where(scf_dict['mo_occ'] >= 1)[0])
    nbeta_total = len(np.where(scf_dict['mo_occ'] >= 2)[0])
    assert nalpha_total + nbeta_total == total_electrons

    print("Num ele active ", active_ne)
    print("Num orbs active ", active_norb)

    print("Total ele ", total_electrons)

    ncore_electrons = total_electrons - active_ne
    assert ncore_electrons % 2 == 0
    print("Num core ", ncore_electrons)
    print("num core alpha ", ncore_electrons // 2)
    print("num core beta ", ncore_electrons // 2)

    ncore_alpha = ncore_electrons // 2
    ncore_beta = ncore_electrons // 2

    active_alpha = int(nalpha_total - ncore_alpha)
    active_beta = int(nbeta_total - ncore_beta)
    print("Num active alpha ", active_alpha)
    print("Num active beta ", active_beta)
    assert np.isclose(active_beta + active_alpha, active_ne)

    if nalpha_total == nbeta_total:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.e_tot = scf_dict['e_tot']
    mf.mo_coeff = scf_dict['mo_coeff']
    mf.mo_occ = scf_dict['mo_occ']
    mf.mo_energy = scf_dict['mo_energy']

    mycas = mcscf.CASSCF(mf, active_norb, (active_alpha, active_beta))
    mo = mycas.sort_mo(occ_orbitals + virt_orbitals, base=0)
    h1e_cas, ecore = mycas.get_h1eff(mo)
    print(h1e_cas.shape, ecore)
    h2e_cas = mycas.get_h2eff(mo)
    print(h2e_cas.shape)
    h2e_cas = ao2mo.restore(1, h2e_cas, h1e_cas.shape[0])
    # now use mo coeffs from avas
    with h5py.File("hamiltonian_{}_{}_{}_spin{}_{}.h5".format(sys_name, scf_type, basis, spin, loc_type), 'w') as fid:
        fid.create_dataset('ecore', data=float(ecore), dtype=float)
        fid.create_dataset('h1', data=h1e_cas)
        fid.create_dataset('eri', data=h2e_cas)
        fid.create_dataset('active_nalpha', data=int(active_alpha), dtype=int)
        fid.create_dataset('active_nbeta', data=int(active_beta), dtype=int)

    with h5py.File("hamiltonian_{}_{}_{}_spin{}_{}.h5".format(sys_name, scf_type, basis, spin, loc_type), 'r') as fid:
        ecore = fid['ecore'][...]
        h1e_cas = fid['h1'][...]
        h2e_cas = fid['eri'][...]
        active_alpha = int(fid['active_nalpha'][...])
        active_beta = int(fid['active_nbeta'][...])
        active_norb = h1e_cas.shape[0]


    h1 = h1e_cas
    eri_full = h2e_cas
    docc = slice(None, min(active_alpha, active_beta))
    socc = slice(min(active_alpha, active_beta), max(active_alpha, active_beta))

    if nalpha_total == nbeta_total:
        scf_energy = ecore + \
                     2*np.einsum('ii',h1[:num_alpha,:num_alpha]) + \
                     2*np.einsum('iijj',eri_full[:num_alpha,:num_alpha,:num_alpha,:num_alpha]) - \
                       np.einsum('ijji',eri_full[:num_alpha,:num_alpha,:num_alpha,:num_alpha])
    else:
        scf_energy = ecore + \
                 2.0 * np.einsum('ii', h1[docc, docc]) + \
                 np.einsum('ii', h1[socc, socc]) + \
                 2.0 * np.einsum('iijj', eri_full[docc, docc, docc, docc]) - \
                 np.einsum('ijji', eri_full[docc, docc, docc, docc]) + \
                 np.einsum('iijj', eri_full[socc, socc, docc, docc]) - \
                 0.5 * np.einsum('ijji', eri_full[socc, docc, docc, socc]) + \
                 np.einsum('iijj', eri_full[docc, docc, socc, socc]) - \
                 0.5 * np.einsum('ijji', eri_full[docc, socc, socc, docc]) + \
                 0.5 * np.einsum('iijj', eri_full[socc, socc, socc, socc]) - \
                 0.5 * np.einsum('ijji', eri_full[socc, socc, socc, socc])
    print(scf_energy, mf.e_tot - mol.energy_nuc(), mf.energy_elec()[0], mf.e_tot)
    assert np.isclose(scf_energy, mf.e_tot)

    # check if active space energy is okay
    active_mol = gto.M()
    active_mol.nelectron = active_ne
    mf_a_rhf = scf.ROHF(active_mol)
    mf_a_rhf.nelec = (active_alpha, active_beta)
    mf_a_rhf.get_hcore = lambda *args: np.asarray(h1e_cas)
    mf_a_rhf.get_ovlp = lambda *args: np.eye(h1e_cas.shape[0])
    mf_a_rhf.energy_nuc = lambda *args: ecore
    mf_a_rhf._eri = h2e_cas
    mf_a_rhf.init_guess = '1e'
    mf_a_rhf.mo_coeff = np.eye(h1e_cas.shape[0])

    alpha_occ = np.array([1] * active_alpha + [0] * (active_norb - active_alpha))
    beta_occ = np.array([1] * active_beta + [0] * (active_norb - active_beta))
    mf_a_rhf.mo_occ = alpha_occ + beta_occ
    w, v = np.linalg.eigh(mf_a_rhf.get_fock())
    mf_a_rhf.mo_energy = w
    print(w)
    mf_a_rhf.e_tot = mf_a_rhf.energy_tot()  # actually calculate the energy
    assert np.isclose(mf.e_tot, mf_a_rhf.e_tot)



if __name__ == "__main__":
    scf_type = 'rohf'
    spin = 5
    basis = 'ccpvdz'
    loc_type = 'pm'
    chkfile_path = 'heme_cys_rohf_ccpvdz_mult6.chk'
    # active_orbs
    occ_orbitals = sorted(
        [74, 75, 77, 83, 92, 93, 95, 96, 97, 98, 99, 103, 104, 82, 88, 105, 106,
         107, 52, 62, 94, 100])
    virt_orbitals = sorted(
        [108, 109, 114, 119, 121, 122, 125, 129, 132, 133, 134, 138, 152, 153,
         154, 155, 167, 259, 137])

    print(chkfile_path)
    active_space_generator(chkfile_path, 'heme_cys', scf_type, spin, basis,
                           loc_type, occ_orbitals, virtual_orbitals)
