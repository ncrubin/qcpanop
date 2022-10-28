"""
Using pyscf's object and Openfermion's objects construct the 2-RDM
from a DOCI calculation and FQE calculation and confirm the blocking structure
"""
from itertools import product

import numpy as np

from pyscf import gto, scf, ao2mo, fci, mcscf

from qcpanop.ci.utils import pyscf_to_fqe_wf

import fqe
from fqe.openfermion_utils import integrals_to_fqe_restricted

import openfermion as of
from openfermion.ops.representations.doci_hamiltonian import DOCIHamiltonian,get_projected_integrals_from_doci, get_tensors_from_doci
from openfermion.chem.molecular_data import spinorb_from_spatial
# def get_doci_opdm(doci_wf, nmo):

from scipy.sparse import coo_matrix




def occ_virt_ordering(h1, eri):
    """
    apply a basis rotation that reorders the orbitals such that at half filling
    we would have the Hartree-Fock state as [1, 0, 1, 0, ...., 1, 0]

    [0, N/2, 1, N/2 + 1, 2, N/2 + 2]

    :param h1:
    :param eri:
    :return:
    """
    norbs = h1.shape[0]
    if norbs % 2 == 1:
        raise ValueError("Only implemented for even number of orbitals right now")
    perm_order = [(xx, (norbs // 2) + xx) for xx in range(norbs//2)]
    perm_order = [k for pair in perm_order for k in pair]
    new_h1 = h1[:, perm_order]
    new_h1 = new_h1[perm_order, :]
    new_eri = eri[:, :, :, perm_order]
    new_eri = new_eri[:, :, perm_order, :]
    new_eri = new_eri[:, perm_order, :, :]
    new_eri = new_eri[perm_order, :, :, :]
    return new_h1, new_eri

def get_doci_fermion_operator(h1, eri):
    """

    :param h1: one-electron spatial MO integrals
    :param eri: two-electron spatial MO integrals in OpenFermion notation <12|2'1'>
    :return:  FermionOperator
    """
    # 2 * t[i, i]n_{i} + sum_{i,j}(2V_{ijij} - V_{ijji}n_{i}n_{j} +
    #  sum_{i=/=j}V_{ii,jj}B_{i}^B_{j}
    # B^_{i} = c_{i,alpha}^c_{i,beta}^  B_{i} = c_{i,beta}c_{i,alpha}
    # n_{i} = B_{i}^B_{i} = c_{i,alpha}^c_{i,beta}^c_{i,beta}c_{i,alpha} = c_{i,alpha}^ n_{i,beta} c_{i,alpha}
    #  c_{i,alpha}^c_{i,beta}^c_{i,beta}c_{i,alpha} = - c_{i,beta}^ c_{i,alpha}^c_{i,beta}c_{i,alpha}
    #  - c_{i,beta}^ c_{i,alpha}^c_{i,beta}c_{i,alpha}  = - c_{i,beta}^ ( delta_{i,alpha, i,beta) - c_{i,beta}c_{i,alpha}^) c_{i,alpha}
    #  c_{i,beta}^c_{i,beta}c_{i,alpha}^c_{i,alpha}   = n_{i,beta)n_{i,alpha}
    def beta_i(i):
        return of.FermionOperator(((i, 1), (i + 1, 1)))
    def n_i(i):
        # return beta_i(i) * of.hermitian_conjugated(beta_i(i))
        # the following return is the same as the comment above...just expanded using CAR
        return of.FermionOperator(((2 * i , 1), (2 * i, 0), (2 * i + 1, 1), (2 * i + 1, 0)))

    chem_eri = eri.transpose(0, 1, 3, 2)
    doci_fermion_op = of.FermionOperator()
    for ii in range(h1.shape[0]):
        doci_fermion_op += 2 * h1[ii, ii] * n_i(ii)
        # n_{ia}n_{ib}
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        doci_fermion_op += (2 * chem_eri[ii, jj, ii, jj] - chem_eri[ii, jj, jj, ii]) * n_i(ii) * n_i(jj)
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        if ii != jj:
            doci_fermion_op += chem_eri[ii, ii, jj, jj] * beta_i(2 * ii) * of.hermitian_conjugated(beta_i(2 * jj))
    return doci_fermion_op


def get_doci_fermion_operator2(h1, eri):
    """

    :param h1: one-electron spatial MO integrals
    :param eri: two-electron spatial MO integrals in OpenFermion notation <12|2'1'>
    :return:  FermionOperator
    """
    # 2 * t[i, i]n_{i} + sum_{i,j}(2V_{ijij} - V_{ijji}n_{i}n_{j} +
    #  sum_{i=/=j}V_{ii,jj}B_{i}^B_{j}
    # B^_{i} = c_{i,alpha}^c_{i,beta}^  B_{i} = c_{i,beta}c_{i,alpha}
    # n_{i} = B_{i}^B_{i} = c_{i,alpha}^c_{i,beta}^c_{i,beta}c_{i,alpha} = c_{i,alpha}^ n_{i,beta} c_{i,alpha}
    #  c_{i,alpha}^c_{i,beta}^c_{i,beta}c_{i,alpha} = - c_{i,beta}^ c_{i,alpha}^c_{i,beta}c_{i,alpha}
    #  - c_{i,beta}^ c_{i,alpha}^c_{i,beta}c_{i,alpha}  = - c_{i,beta}^ ( delta_{i,alpha, i,beta) - c_{i,beta}c_{i,alpha}^) c_{i,alpha}
    #  c_{i,beta}^c_{i,beta}c_{i,alpha}^c_{i,alpha}   = n_{i,beta)n_{i,alpha}
    def beta_i(i):
        return of.FermionOperator(((i, 1), (i + 1, 1)))
    def n_i(i):
        # return beta_i(i) * of.hermitian_conjugated(beta_i(i))
        # the following return is the same as the comment above...just expanded using CAR
        # return of.FermionOperator(((2 * ii , 1), (2 * ii, 0), (2 * ii + 1, 1), (2 * ii + 1, 0)))
        return of.FermionOperator(((2 * i, 1), (2 * i, 0))) + of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)))

    chem_eri = eri.transpose(0, 1, 3, 2)
    doci_fermion_op = of.FermionOperator()
    for ii in range(h1.shape[0]):
        doci_fermion_op += h1[ii, ii] * n_i(ii)
        # n_{ia}n_{ib}
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        if ii != jj:
            doci_fermion_op += 0.25 * (2 * chem_eri[ii, jj, ii, jj] - chem_eri[ii, jj, jj, ii]) * n_i(ii) * n_i(jj)
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        doci_fermion_op += chem_eri[ii, ii, jj, jj] * beta_i(2 * ii) * of.hermitian_conjugated(beta_i(2 * jj))
    return doci_fermion_op


def get_doci_qubit_operator(h1, eri):
    """
    s+ = 0.5 (X - 1j Y)
    s- = 0.5 (X + 1j Y)
    [s-, s+] = Z
    [Z, s-] = 2s-
    [Z, s+] = -2s+

    Recall:
    [P_i, P_j^] = (1 - N_i)delta_{ij}
    [1 - N_i, P_i] = 2 P_i
    [1 - N_i, P_i^] = -2 P_i^

    Therefore, we can see that (1 - N_i) = Z, P_i = s-,  P_i^ = s+

    Thereffore,
    1 - N = Z, 1 - Z - N = 0, 1 - Z = N

    Recall The S0 Hamiltonian is

    sum_p h_p N_p + 0.25 * sum_{p =\= q} w_{p,q}N_p N_q + sum_pq v_pq P_{p}^ P_{q}

    which is converted to
    sum_p h_p N_p + 0.25 * sum_{p =\= q} w_{p,q}N_p N_q + sum_pq v_pq P_{p}^ P_{q}
    sum_p h_p (1 - Z_p) + 0.25 * sum_{p =\= q} w_{pq}(1 - Z_p)(1 - Z_q) + ...

    (1 - Z_p)(1 - Z_q) = (1 - Z_p - Z_q + Z_pZ_q)

    If Z = 1 - N_i
    N|11> = 2 |11>,  N|00> = 0|00>
    (1 - N_i)|11> = |11> - 2|11> = -|11>,  (1 - N_i)|00> = |00>
    So it looks like the normal Z operator on the pair space.
    Z|11> = -|11>, Z|00> = |00>
    (1 - Z)|11> = |11> - -|11> = 2|11>  (1 - Z)|00> = |00> - |00> = 0|00>

    Sp|00> = |11>, Sp|11> = 0|11>
    Sm|11> = |00>, Sm|11> = |00>

    :param h1: one-electron spatial MO integrals
    :param eri: two-electron spatial MO integrals in OpenFermion notation <12|2'1'>
    :return:  QubitOperator
    """
    sp = lambda x: of.QubitOperator(((x, 'X')), coefficient=0.5) - of.QubitOperator(((x, 'Y')),coefficient=0.5j)
    sm = lambda x: of.QubitOperator(((x, 'X')), coefficient=0.5) + of.QubitOperator(((x, 'Y')),coefficient=0.5j)
    def beta_i(i):
        return sp(i)
    def n_i(i):
        # return beta_i(i) * of.hermitian_conjugated(beta_i(i))
        # the following return is the same as the comment above...just expanded using CAR
        # return of.FermionOperator(((2 * ii , 1), (2 * ii, 0), (2 * ii + 1, 1), (2 * ii + 1, 0)))
        return 1 - of.QubitOperator(((i, 'Z')))

    chem_eri = eri.transpose(0, 1, 3, 2)
    doci_qubit_op = of.QubitOperator()
    for ii in range(h1.shape[0]):
        doci_qubit_op += h1[ii, ii] * n_i(ii)
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        if ii != jj:
            doci_qubit_op += 0.25 * (2 * chem_eri[ii, jj, ii, jj] - chem_eri[ii, jj, jj, ii]) * n_i(ii) * n_i(jj)
    for ii, jj in product(range(h1.shape[0]), repeat=2):
        doci_qubit_op += chem_eri[ii, ii, jj, jj] * sp(ii) * sm(jj)
    return doci_qubit_op


def get_doci_from_integrals(one_body_integrals, two_body_integrals):
    r"""Construct a DOCI Hamiltonian from electron integrals

    Args:
        one_body_integrals [numpy array]: one-electron integrals
        two_body_integrals [numpy array]: two-electron integrals

    Returns:
        hc [numpy array]: The single-particle DOCI terms in matrix form
        hr1 [numpy array]: The off-diagonal DOCI Hamiltonian terms in matrix
            form
        hr2 [numpy array]: The diagonal DOCI Hamiltonian terms in matrix form
    """

    n_qubits = one_body_integrals.shape[0]
    hc = np.zeros(n_qubits)
    hr1 = np.zeros((n_qubits, n_qubits))
    hr2 = np.zeros((n_qubits, n_qubits))

    for p in range(n_qubits):
        hc[p] = 2 * one_body_integrals[p, p]
        for q in range(n_qubits):
            hr2[p, q] = (2 * two_body_integrals[p, q, q, p] -
                         two_body_integrals[p, q, p, q])
            if p == q:
                continue
            hr1[p, q] = two_body_integrals[p, p, q, q]

    return hc, hr1, hr2


def doci_wf_to_cirq_wf(doci_wf, nmo):
    cirq_wf = np.zeros((4**nmo, 1), dtype=np.complex128)
    for ii in range(2**nmo):
        doci_ket = np.binary_repr(ii, width=nmo)
        cirq_idx = "".join([j for i in zip(doci_ket, doci_ket) for j in i])
        if not np.isclose(doci_wf[ii], 0):
            print(doci_ket, cirq_idx, doci_wf[ii])
        cirq_wf[int(cirq_idx, 2)] = doci_wf[ii]
    return cirq_wf


def get_mo_via_cas(mf):
    cas = mcscf.CASCI(mf, ncas=mf.mo_coeff.shape[1], nelecas=mf.mol.nelectron)
    h1, ecore = cas.get_h1eff()
    eri = cas.get_h2cas()
    h1 = np.ascontiguousarray(h1)
    eri = np.ascontiguousarray(eri)
    eri = ao2mo.restore('s1', eri, h1.shape[0])
    ecore = float(ecore)

    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    tei = np.asarray(
        eri.transpose(0, 2, 3, 1), order='C')
    ele_ham = integrals_to_fqe_restricted(h1, tei)
    ele_ham._e_0 = ecore
    return ele_ham, ecore, h1, tei


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

def get_doci_basis_to_full_basis(nspatial):
    """nspatial ends up being the number of qubits"""
    doci_to_fullspace = dict()
    for ii in range(2**nspatial):
        ket = np.binary_repr(ii, width=nspatial)
        full_space_ket = [i for sub in zip(ket, ket) for i in sub]
        doci_to_fullspace[ii] = int("".join(full_space_ket), 2)
    return doci_to_fullspace


def check_reordered_orb_hf_energy(mf):
    fake_mol = gto.M()
    fake_mol.nelectron = mf.mol.nelectron

    ele_ham, ecore, h1, eri = get_mo_via_cas(mf)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    tei = np.asarray(
        eri.transpose(0, 3, 1, 2), order='C')
    num_alpha = mf.mol.nelectron // 2

    # Check the SCF Energy
    scf_energy = ecore + \
                 2 * np.einsum('ii', h1[:num_alpha, :num_alpha]) + \
                 2 * np.einsum('iijj', tei[:num_alpha, :num_alpha, :num_alpha, :num_alpha]) - \
                 np.einsum('ijji', tei[:num_alpha, :num_alpha, :num_alpha, :num_alpha])
    assert np.isclose(scf_energy, mf.e_tot)

    reordered_h1, reordered_tei = occ_virt_ordering(h1, tei)
    mf_hf = scf.RHF(fake_mol)
    mf_hf.get_hcore = lambda *args: np.asarray(reordered_h1)
    mf_hf.get_ovlp = lambda *args: np.eye(h1.shape[0])
    mf_hf.energy_nuc = lambda *args: mf.energy_nuc()
    mf_hf._eri = reordered_tei # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)
    mf_hf.init_guess = '1e'
    mf_hf.mo_occ = np.array([2, 0, 2, 0, 0, 0])
    mf_hf.mo_coeff = np.eye(h1.shape[0])

    assert np.isclose(mf_hf.energy_elec()[0], mf.energy_elec()[0])
    return mf_hf

def check_reordered_orb_fci_energy(reord_mf, original_fci_obj):
    cisolver = fci.FCI(reord_mf)
    fci_e, fci_civec = cisolver.kernel()
    wf = pyscf_to_fqe_wf(fci_civec, pyscf_mf=reord_mf)
    fci_fqe_wf = wf
    ele_ham, ecore, h1, eri = get_mo_via_cas(reord_mf)
    assert np.isclose(fci_e, fci_fqe_wf.expectationValue(ele_ham).real)
    assert np.isclose(original_fci_obj.e_tot, fci_e)

def main():
    # Build molecule and get SCF, FCI solution
    mol = gto.M()
    # mol.atom = [
    #     ['C',    [-0.7949418331, -0.7689422790,  0.0000431796]],
    #     ['C',    [ 0.7942967036, -0.7689422790, -0.0000431723]],
    #     ['C',    [-0.6722273179,  0.7688327240, -0.0000510161]],
    #     ['C',    [ 0.6728724474,  0.7690518340,  0.0000510089]],
    #     ['H',    [-1.2552274809, -1.2127153191,  0.9009251083]],
    #     ['H',    [-1.2554161144, -1.2129497751, -0.9006212193]],
    #     ['H',    [ 1.2544214424, -1.2128841657, -0.9009060529]],
    #     ['H',    [ 1.2546085255, -1.2131181371,  0.9006033449]],
    #     ['H',    [-1.4365790690,  1.5570891855, -0.0001303938]],
    #     ['H',    [ 1.4374598126,  1.5571331655,  0.0001299351]],]
    mol.atom = [['Li', [0, 0, 0]],
                ['H',  [0, 0, 1.6]]]
    # mol.atom = [['N', [0, 0, 0]],
    #             ['N',  [0, 0, 1.6]]]
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.run()
    nmo = mf.mo_coeff.shape[1]

    reorderd_mf = check_reordered_orb_hf_energy(mf)

    print(nmo, mol.nelectron)
    cisolver = fci.FCI(mf)
    fci_e, fci_civec = cisolver.kernel()
    fci_fqe_wf = pyscf_to_fqe_wf(fci_civec, pyscf_mf=mf)
    ele_ham, ecore, h1, eri = get_mo_via_cas(mf)
    assert np.isclose(fci_e, fci_fqe_wf.expectationValue(ele_ham).real)
    check_reordered_orb_fci_energy(reorderd_mf, cisolver)


    # Project Molecular Spin Hamiltonian into DOCI sector
    # and diagonalize to confirm doci energy
    spin_oei, spin_tei = spinorb_from_spatial(h1, eri)
    aspin_tei = 0.25 * (spin_tei - np.einsum('ijlk', spin_tei))
    mol_ham = of.InteractionOperator(ecore, spin_oei, aspin_tei)
    print("Getting sparse ham operator")
    dense_mol_ham = of.get_sparse_operator(mol_ham).toarray()
    w, v = np.linalg.eigh(dense_mol_ham)
    print("Checking energy")
    assert np.isclose(w[0], fci_e)

    print("Building projector and S0 basis")
    seniority_zero_projector, seniority_zero_basis = get_doci_projector(2 * nmo,
                                                                        mol.nelectron)
    print("projecting Hamiltoniann")
    # zero out everything but seniority zero sector
    doci_ham_test = seniority_zero_projector @ dense_mol_ham @ seniority_zero_projector
    # take just the seniority zero basis
    doci_ham_test = doci_ham_test[:, seniority_zero_basis]
    doci_ham_test = doci_ham_test[seniority_zero_basis, :]
    # solve in the S0 determinant space
    # we do this for numerical stability
    print("DIagonalizing Hamiltonian")
    proj_doci_eigs, proj_doci_vecs = np.linalg.eigh(doci_ham_test)
    # put wavefuction on S0 space back into full space
    full_space_wf = np.zeros((4**nmo), dtype=np.complex128)
    for idx in range(len(seniority_zero_basis)):
        full_space_wf[seniority_zero_basis[idx]] = proj_doci_vecs[idx, 0]
    # print wf
    fqe_doci = fqe.from_cirq(full_space_wf.flatten(), thresh=1.0E-12)
    fqe_doci.print_wfn()
    # check S0 wf gives the same expectation value
    doci_energy = fqe_doci.expectationValue(ele_ham)
    print(doci_energy)
    assert np.isclose(doci_energy, proj_doci_eigs[0])

    fci_opdm, fci_tpdm = fqe_doci.sector((mol.nelectron, 0)).get_openfermion_rdms()
    print(fci_opdm)
    exit()

    ############################################################################
    #
    #  Now that we have checked the energy and projection and going to and
    #  from the S0 space we should check the OpenFermion DOCI code. I suspect
    #  there might be something wrong there.  The thing to check is if the
    #  full hamiltonian -> DOCI Hamiltonian has an identical spectrum to
    #  the projected Hamiltonian we used above. We also want to check that the
    #  RDMs can be reconstructed and we have all the appropriate symmetries.
    #
    ############################################################################

    # Get DOCI hamiltonian using code in OpenFermion
    hc, doci_h1, doci_eri = get_doci_from_integrals(h1, eri)
    doci_ham = DOCIHamiltonian(ecore, hc, doci_h1, doci_eri)
    w, v = np.linalg.eigh(of.get_sparse_operator(doci_ham.qubit_operator).toarray())
    gs_e = w[0]
    gs_wf = v[:, [0]]
    for ii in range(2**nmo):
        if not np.isclose(gs_wf[ii], 0):
            print(np.binary_repr(ii, width=nmo), gs_wf[ii])
    print("current openfermion doci spectrum: first 10")
    print(w[:10])
    ncr_doci_fop = get_doci_fermion_operator2(h1=h1, eri=eri)
    ncr_doci_ham = of.get_sparse_operator(ncr_doci_fop).toarray().real
    ncr_doci_ham = ncr_doci_ham[:, seniority_zero_basis]
    ncr_doci_ham = ncr_doci_ham[seniority_zero_basis, :]
    ncr_w, ncr_v = np.linalg.eigh(ncr_doci_ham)
    print("Nicks doci spectrum: first 10")
    print(ncr_w[:10])
    full_space_wf = np.zeros((4**nmo), dtype=np.complex128)
    for idx in range(len(seniority_zero_basis)):
        full_space_wf[seniority_zero_basis[idx]] = ncr_v[idx, 0]
    fqe_doci = fqe.from_cirq(full_space_wf.flatten(), thresh=1.0E-12)
    fqe_doci.print_wfn()
    # check S0 wf gives the same expectation value
    doci_energy = fqe_doci.expectationValue(ele_ham)
    print("Doci energy against FQE Hamiltonain")
    print(doci_energy)


    ncr_doci_qop = get_doci_qubit_operator(h1=h1, eri=eri)
    ncr_wq, ncr_vq = np.linalg.eigh(of.get_sparse_operator(ncr_doci_qop).toarray().real)
    print("DOCI qubit operator Hamiltonian")
    print(ncr_wq[:10] + ecore)
    for ii in range(2**nmo):
        if not np.isclose(ncr_vq[ii, 0], 0):
            print(np.binary_repr(ii, width=nmo), ncr_vq[ii, 0])
    full_space_wf = np.zeros((4 ** nmo), dtype=np.complex128)
    for idx in range(2**nmo):
        ket_s0 = np.binary_repr(idx, width=nmo)
        full_space_ket = "".join([i for sub in zip(ket_s0, ket_s0) for i in sub])
        # print(ket_s0)
        # print(full_space_ket)
        # print(int(full_space_ket, 2))
        # print()
        full_space_wf[int(full_space_ket, 2)] = ncr_vq[idx, 0]

    # print wf
    fqe_doci = fqe.from_cirq(full_space_wf.flatten(), thresh=1.0E-12)
    fqe_doci.print_wfn()
    # check S0 wf gives the same expectation value
    doci_energy = fqe_doci.expectationValue(ele_ham)
    print(doci_energy)

    print(ncr_doci_qop)

if __name__ == "__main__":
    main()