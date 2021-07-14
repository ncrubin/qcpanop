import numpy as np
from numpy import einsum


def ccd_energy(t2, h, g, o, v):
    energy = 0.
    #	  1.0000 h(i,i)
    energy += 1.0 * einsum('ii', h[o, o])

    #	  0.5000 <i,j||i,j>
    energy += 0.5 * einsum('ijij', g[o, o, o, o])

    #	  0.2500 <i,j||a,b>*t2(a,b,i,j)
    energy += 0.25 * einsum('ijab,abij', g[o, o, v, v], t2)
    return energy


def doubles_residual_contractions(t2, g, o, v):
    doubles_residual = np.zeros_like(t2)
    #	  1.0000 <e,f||m,n>
    doubles_residual += 1.0 * einsum('efmn->efmn', g[v, v, o, o])

    #	  0.5000 <i,j||m,n>*t2(e,f,i,j)
    doubles_residual += 0.5 * einsum('ijmn,efij->efmn', g[o, o, o, o], t2)

    #	 -1.0000 <i,e||a,n>*t2(a,f,i,m)
    doubles_residual += -1.0 * einsum('iean,afim->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,e||a,m>*t2(a,f,i,n)
    doubles_residual += 1.0 * einsum('ieam,afin->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,f||a,n>*t2(a,e,i,m)
    doubles_residual += 1.0 * einsum('ifan,aeim->efmn', g[o, v, v, o], t2)

    #	 -1.0000 <i,f||a,m>*t2(a,e,i,n)
    doubles_residual += -1.0 * einsum('ifam,aein->efmn', g[o, v, v, o], t2)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_residual += 0.5 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)

    #	 -0.5000 <i,j||a,b>*t2(a,b,n,j)*t2(e,f,m,i)
    doubles_residual += -0.5 * einsum('ijab,abnj,efmi->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,b,m,j)*t2(e,f,n,i)
    doubles_residual += 0.5 * einsum('ijab,abmj,efni->efmn', g[o, o, v, v], t2,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <i,j||a,b>*t2(a,b,m,n)*t2(e,f,i,j)
    doubles_residual += 0.25 * einsum('ijab,abmn,efij->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,e,i,j)*t2(b,f,m,n)
    doubles_residual += -0.5 * einsum('ijab,aeij,bfmn->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t2(a,e,n,j)*t2(b,f,m,i)
    doubles_residual += 1.0 * einsum('ijab,aenj,bfmi->efmn', g[o, o, v, v], t2,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t2(a,e,m,j)*t2(b,f,n,i)
    doubles_residual += -1.0 * einsum('ijab,aemj,bfni->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,e,m,n)*t2(b,f,i,j)
    doubles_residual += -0.5 * einsum('ijab,aemn,bfij->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 2), (0, 1)])
    return doubles_residual


def kernel(t2, h, g, o, v, e_abij, max_iter=100, stopping_eps=1.0E-8):

    old_energy = ccd_energy(t2, h, g, o, v)
    for idx in range(max_iter):

        doubles_res = doubles_residual_contractions(t2, g, o, v)

        new_doubles = doubles_res * e_abij

        current_energy = ccd_energy(new_doubles, h, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            return new_doubles
        else:
            t2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        print("Did not converge")
        return new_doubles

def run_ccd_from_molecule(molecule):
    """Run CCD and return amplitudes"""
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    nso = 2 * norbs
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    nvirt = norbs - nocc

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    pyscf_astei = np.einsum('ijlk', stei)
    pyscf_astei = pyscf_astei - np.einsum('ijlk', pyscf_astei)

    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    t1s = spatial2spin(mycc.t1)
    t2s = spatial2spin(mycc.t2)
    h = soei
    g = gtei
    t2 = t2s.transpose(2, 3, 0, 1)

    t2f = kernel(np.zeros((nso - nele, nso - nele, nele, nele)), h, g, o, v, e_abij)
    return t2f


def main():
    from itertools import product
    import pyscf
    import openfermion as of
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np
    from scipy.linalg import expm

    import fqe
    from fqe.openfermion_utils import molecular_data_to_restricted_fqe_op

    from openfermion.chem.molecular_data import spinorb_from_spatial

    basis = 'cc-pvdz'
    mol = pyscf.M(
        atom='H 0 0 0; B 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF().run()
    mycc = pyscf.cc.CCSD(mf)
    # mycc.frozen = 1
    # old_update_amps = mycc.update_amps

    # def update_amps(t1, t2, eris):
    #     t1, t2 = old_update_amps(t1, t2, eris)
    #     return t1 * 0, t2

    # mycc.update_amps = update_amps
    # mycc.kernel()

    # print('CCD correlation energy', mycc.e_corr)
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 1.6)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_ccsd=True)
    hamiltonian = molecule.get_molecular_hamiltonian()
    elec_ham = molecular_data_to_restricted_fqe_op(molecule)
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    nso = 2 * norbs
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    nvirt = norbs - nocc
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)
    print("nocc ", 2 * nocc)
    print('nvirt ', nso - 2 * nocc)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    pyscf_astei = np.einsum('ijlk', stei)
    pyscf_astei = pyscf_astei - np.einsum('ijlk', pyscf_astei)

    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    print(np.einsum('ii', fock[o, o]), np.einsum('ii', (soei + 0.5 * np.einsum('piiq->pq', astei[:, o, o, :]))[o, o]), hf_energy)
    exit()
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    t1s = spatial2spin(mycc.t1)
    t2s = spatial2spin(mycc.t2)
    h = soei
    g = gtei
    t2 = t2s.transpose(2, 3, 0, 1)
    print(t2.shape)
    print(np.zeros((nso - nele, nele)).shape)

    t2f = kernel(np.zeros((nso - nele, nso - nele, nele, nele)), h, g, o, v, e_abij)
    print(ccd_energy(t2f, h, g, o, v) - hf_energy)


if __name__ == "__main__":
    main()