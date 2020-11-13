"""
Implement RHF equations with DIIS and other forms of optimization
"""
import numpy as np
import scipy as sp
from diis import DIIS


class RHF:

    def __init__(self, hcore, overlap, eri, nelectrons, iter_max=50,
                 rmsd_eps=1.0E-8, diis_length=6):
        self.hcore = hcore
        self.overlap = overlap
        self.eri = eri  # (ij|kl)
        self.nelec = nelectrons

        self.nocc = int(self.nelec // 2)
        self.occ = list(range(self.nocc))

        self.iter_max = iter_max
        self.rmsd_eps = rmsd_eps
        self.diis_length = diis_length

    def core_density(self):
        _, c = sp.linalg.eigh(self.hcore, b=self.overlap)
        return 2 * (c[:, self.occ] @ c[:, self.occ].T)

    def j_mat(self, dmat):
        return np.einsum('ijkl,kl', self.eri, dmat)

    def k_mat(self, dmat):
        return np.einsum('ilkj,kl', self.eri, dmat)

    def fock_mat(self, dmat):
        return self.hcore + self.j_mat(dmat) - 0.5 * self.k_mat(dmat)

    def new_density_from_fock(self, fock):
        _, c = sp.linalg.eigh(fock, b=self.overlap)
        return 2 * (c[:, self.occ] @ c[:, self.occ].T)

    def solve(self):
        np.set_printoptions(linewidth=300)
        iter = 0
        dmat = self.core_density()
        d_old = dmat.copy()
        e_old = np.trace(self.hcore @ dmat)
        rmsd = 10
        print("Iter\tE(elec)\tdE\tRMSD\n")
        while iter < self.iter_max and rmsd > self.rmsd_eps:
            fock = self.fock_mat(dmat)
            error = fock @ dmat @ self.overlap - self.overlap @ dmat @ fock
            dmat = self.new_density_from_fock(fock)
            current_e = 0.5 * np.einsum('ij,ij', self.hcore + fock, dmat)
            dE = abs(current_e - e_old)
            rmsd = np.linalg.norm(dmat - d_old)
            rmsd = np.linalg.norm(error)

            # print(error)
            print("{}\t{: 5.10f}\t{: 5.5f}\t{: 5.5f}".format(iter, current_e, dE,
                                                            rmsd))
            d_old = dmat.copy()
            e_old = current_e
            iter += 1
        e, v = sp.linalg.eigh(fock, b=self.overlap)
        self.dmat = dmat
        self.fock = fock
        self.mo_coeff = v
        self.mo_energies = e

    def solve_diis(self):
        diis = DIIS(self.diis_length)
        iter = 0
        dmat = self.core_density()
        d_old = dmat.copy()
        e_old = np.trace(self.hcore @ dmat)
        rmsd = 10
        print("Iter\tE(elec)\tdE\tRMSD\n")
        while iter < self.iter_max and rmsd > self.rmsd_eps:
            fock = self.fock_mat(dmat)
            error = fock @ dmat @ self.overlap - self.overlap @ dmat @ fock
            fock = diis.compute_new_vec(fock, error)

            dmat = self.new_density_from_fock(fock)
            current_e = 0.5 * np.einsum('ij,ij', self.hcore + fock, dmat)
            dE = abs(current_e - e_old)
            rmsd = np.linalg.norm(dmat - d_old)

            # print(error)
            print("{}\t{: 5.10f}\t{: 5.5f}\t{: 5.5f}".format(iter, current_e, dE,
                                                            rmsd))
            d_old = dmat.copy()
            e_old = current_e
            iter += 1

        e, v = sp.linalg.eigh(fock, b=self.overlap)
        self.dmat = dmat
        self.fock = fock
        self.mo_coeff = v
        self.mo_energies = e


if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    from pyscf import gto, scf
    import openfermion as of

    mol = gto.M(
        verbose=0,
        atom='O   0.000000000000  -0.143225816552   0.000000000000;H  1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000',
        basis='sto-3g',
    )
    # mol = gto.M(
    #     verbose=0,
    #     atom='Li 0 0 0; H 0 0 5.0',
    #     basis='sto-3g',
    # )
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    rhf = RHF(t + v, s, eri, mol.nelectron, iter_max=300,
              diis_length=4)
    rhf.solve_diis()

    print(np.trace(rhf.dmat @ rhf.overlap))
    fock = rhf.fock_mat(rhf.dmat)
    current_e = 0.5 * np.einsum('ij,ij', rhf.hcore + fock, rhf.dmat)
    print(current_e + mol.energy_nuc())

    print(rhf.mo_energies)

    # of_eri = eri.transpose(0, 2, 3, 1)
    # of.general_basis_change(of_eri, )

    rho0 = rhf.core_density()
    mf = scf.RHF(mol)
    mf.diis_space = 6
    mf.kernel(rho0)
    dmat = mf.make_rdm1()
    print(mf.energy_tot())

    fock = rhf.fock_mat(dmat)
    current_e = 0.5 * np.einsum('ij,ij', rhf.hcore + fock, dmat)
    print(current_e, mf.e_tot - mol.energy_nuc())
    print(of.general_basis_change(fock, mf.mo_coeff, (1, 0)))

