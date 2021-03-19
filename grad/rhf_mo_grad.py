"""
Test implementation of Helgaker's MO gradients
"""
from pyscf import gto, scf, ao2mo
import numpy as np
from gradient_integrals import (hcore_generator, overlap_generator,
                                eri_generator, grad_nuc)
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial


def gradient_mo(mol, mo_coeffs, hcore_mo, tei_mo, opdm, tpdm):
    """
    Obtain the gradient given that stationarity has been obtained

    :param mol: pyscf.Mole object for getting AO integrals
    :param mo_coeffs: AO-to-MO molecular orbital coefficients
    :param hcore_mo: Core-MO matrix (n-spatial dim x n-spatial dim)
    :param tei_mo: ERI-MO tensor (n-spatial, n-spatial, n-spatial, n-spatial)
    :param opdm:  one-RDM (in SPIN ORBITALS openFermion ordering)
    :param tpdm: two-RDM (in SPIN ORBITALS openFermion ordering)
    :return: N x 3 matrix where N is the number of atoms. Each row is the force
             vector.
    """
    hcore_deriv = hcore_generator(mol)
    ovrlp_deriv = overlap_generator(mol)
    eri_deriv = eri_generator(mol)
    atmlst = range(mol.natm)
    de = np.zeros((len(atmlst), 3))
    spin_summed_dm = opdm[::2, ::2] + opdm[1::2, 1::2]
    zero_oei = np.zeros_like(hcore_mo)

    for k, ia in enumerate(atmlst):
        h1ao = hcore_deriv(ia)
        s1ao = ovrlp_deriv(ia)
        eriao = eri_deriv(ia)
        h1mo = np.zeros_like(h1ao)
        s1mo = np.zeros_like(s1ao)
        erimo = np.zeros_like(eriao)

        # X-Core-MO - Hellmann-Feynman term
        h1mo[0] = of.general_basis_change(h1ao[0], mo_coeffs, key=(1, 0))
        # Y-Core-MO - Hellmann-Feynman term
        h1mo[1] = of.general_basis_change(h1ao[1], mo_coeffs, key=(1, 0))
        # Z-Core-MO - Hellmann-Feynman term
        h1mo[2] = of.general_basis_change(h1ao[2], mo_coeffs, key=(1, 0))

        # X-S-MO
        s1mo[0] = of.general_basis_change(s1ao[0], mo_coeffs, key=(1, 0))
        # Y-S-MO
        s1mo[1] = of.general_basis_change(s1ao[1], mo_coeffs, key=(1, 0))
        # Z-S-MO
        s1mo[2] = of.general_basis_change(s1ao[2], mo_coeffs, key=(1, 0))

        # first part of wavefunction force
        h1mo[0] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[0]) +
                          np.einsum('ip,jp->ij', hcore_mo, s1mo[0]))
        h1mo[1] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[1]) +
                          np.einsum('ip,jp->ij', hcore_mo, s1mo[1]))
        h1mo[2] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[2]) +
                          np.einsum('ip,jp->ij', hcore_mo, s1mo[2]))

        de[k] += np.einsum('xij,ij->x', h1mo, spin_summed_dm)

        # eriao in openfermion ordering  Hellmann-Feynmen term
        erimo[0] -= of.general_basis_change(eriao[0], mo_coeffs, key=(1, 0, 1, 0)).transpose((0, 2, 3, 1))
        erimo[1] -= of.general_basis_change(eriao[1], mo_coeffs, key=(1, 0, 1, 0)).transpose((0, 2, 3, 1))
        erimo[2] -= of.general_basis_change(eriao[2], mo_coeffs, key=(1, 0, 1, 0)).transpose((0, 2, 3, 1))

        # second part of wavefunction force
        erimo[0] += 0.5 * (np.einsum('px,xqrs', s1mo[0], tei_mo) +
                           np.einsum('qx,pxrs', s1mo[0], tei_mo) +
                           np.einsum('rx,pqxs', s1mo[0], tei_mo) +
                           np.einsum('sx,pqrx', s1mo[0], tei_mo))

        erimo[1] += 0.5 * (np.einsum('px,xqrs', s1mo[1], tei_mo) +
                           np.einsum('qx,pxrs', s1mo[1], tei_mo) +
                           np.einsum('rx,pqxs', s1mo[1], tei_mo) +
                           np.einsum('sx,pqrx', s1mo[1], tei_mo))

        erimo[2] += 0.5 * (np.einsum('px,xqrs', s1mo[2], tei_mo) +
                           np.einsum('qx,pxrs', s1mo[2], tei_mo) +
                           np.einsum('rx,pqxs', s1mo[2], tei_mo) +
                           np.einsum('sx,pqrx', s1mo[2], tei_mo))

        # FIX Later to generate a gradient operator instead.
        de[k][0] += np.einsum('ijkl,ijkl', 0.5 * spinorb_from_spatial(zero_oei, erimo[0])[1],
                    tpdm).real
        de[k][1] += np.einsum('ijkl,ijkl', 0.5 * spinorb_from_spatial(zero_oei, erimo[1])[1],
                    tpdm).real
        de[k][2] += np.einsum('ijkl,ijkl', 0.5 * spinorb_from_spatial(zero_oei, erimo[2])[1],
                    tpdm).real

    de += grad_nuc(mol, atmlst=atmlst)
    return de



if __name__ == "__main__":
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    mol = gto.M(
        verbose=0,
        atom='Li 0 0 0; H 0 0 1.5',
        basis='sto-3g',
    )

    # mol = gto.M(
    #     verbose=0,
    #     atom='O   0.000000000000  -0.143225816552   0.000000000000;H  1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000',
    #     basis='6-31g',
    # )


    mf = scf.RHF(mol)
    mf.kernel()

    molecule = of.MolecularData(geometry=mol.atom, basis=mol.basis,
                                charge=0, multiplicity=1)
    molecule = run_pyscf(molecule)
    oei, tei = molecule.get_integrals()

    nelec = sum(mol.nelec)
    norbs = mf.mo_coeff.shape[1]

    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    ao_hcore = t + v

    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    two_electron_compressed = ao2mo.kernel(mol, mf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed,norbs)
    mo_eri = of.general_basis_change(eri, mf.mo_coeff, key=(1, 0, 1, 0))
    assert np.allclose(mo_eri, two_electron_integrals)

    mo_hcore = of.general_basis_change(ao_hcore, mf.mo_coeff, key=(1, 0))
    assert np.allclose(mo_hcore, oei)

    mo_of_eri = mo_eri.transpose((0, 2, 3, 1))
    assert np.allclose(mo_of_eri, tei)

    spin_hcore, spin_eri = spinorb_from_spatial(mo_hcore, mo_of_eri)

    ham = of.generate_hamiltonian(oei, tei, 0)

    opdm = np.diag([1] * nelec + [0] * (2 * norbs - nelec))
    tpdm = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))
    rdms = of.InteractionRDM(opdm, tpdm)

    from rhf import RHF
    from rhf_gradients import gradient
    rhf = RHF(t + v, s, eri, mol.nelectron, iter_max=300,
              diis_length=4)
    rhf.solve_diis()
    print("True Grad")
    print(gradient(rhf, mol))

    print("Test grad")
    de = gradient_mo(mol, mf.mo_coeff, oei, tei, opdm, tpdm)
    print(de)

