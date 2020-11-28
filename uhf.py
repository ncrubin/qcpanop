"""
Implement UHF equations with DIIS and other forms of optimization
"""
import numpy as np
import scipy as sp
from diis import DIIS
from fon import FON


class UHF:

    def __init__(self, hcore, overlap, eri, nalpha, nbeta, iter_max=50,
                 rmsd_eps=1.0E-8, diis_length=6):
        self.hcore = hcore
        self.overlap = overlap
        self.eri = eri  # (ij|kl)
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.norbs = hcore.shape[0]

        self.occa = list(range(self.nalpha))
        self.occb = list(range(self.nbeta))

        self.iter_max = iter_max
        self.rmsd_eps = rmsd_eps
        self.diis_length = diis_length

    def core_density(self):
        _, c = sp.linalg.eigh(self.hcore, b=self.overlap)
        rho_a = c[:, self.occa] @ c[:, self.occa].T
        rho_b = c[:, self.occb] @ c[:, self.occb].T
        return rho_a, rho_b

    def j_mat(self, rho_t):
        return np.einsum('ijkl,kl', self.eri, rho_t)

    def k_mat(self, rho_spin):
        return np.einsum('ilkj,kl', self.eri, rho_spin)

    def fock_mat(self, rho_a, rho_b):
        f_a = self.hcore + self.j_mat(rho_a + rho_b)
        f_b = f_a.copy()
        f_a -= self.k_mat(rho_a)
        f_b -= self.k_mat(rho_b)
        return f_a, f_b

    def new_density_from_fock(self, fock_a, fock_b):
        _, ca = sp.linalg.eigh(fock_a, b=self.overlap)
        _, cb = sp.linalg.eigh(fock_b, b=self.overlap)
        rho_a = ca[:, self.occa] @ ca[:, self.occa].T
        rho_b = cb[:, self.occb] @ cb[:, self.occb].T
        return rho_a, rho_b

    def fon_density_from_fock(self, fock_a, fock_b, fon_obj, temperature):
        ea, ca = sp.linalg.eigh(fock_a, b=self.overlap)
        eb, cb = sp.linalg.eigh(fock_b, b=self.overlap)
        n_ia = fon_obj.frac_occ_rhf(temperature, ea, len(self.occa))
        n_ib = fon_obj.frac_occ_rhf(temperature, eb, len(self.occb))
        rho_a = np.zeros_like(fock_a)
        rho_b = np.zeros_like(fock_b)
        for idx, (nna, nnb) in enumerate(zip(n_ia, n_ib)):
            rho_a += nna * ca[:, [idx]] @ ca[:, [idx]].T
            rho_b += nnb * cb[:, [idx]] @ cb[:, [idx]].T
        return rho_a, rho_b

    def solve(self):
        np.set_printoptions(linewidth=300)
        iter = 0
        dmat_a, dmat_b = self.core_density()
        da_old = dmat_a.copy()
        db_old = dmat_b.copy()
        e_old = np.trace(self.hcore @ (dmat_a + dmat_b))
        rmsd = 10
        print("Iter\tE(elec)\tdE\tRMSD\n")
        while iter < self.iter_max and rmsd > self.rmsd_eps:
            f_a, f_b = self.fock_mat(dmat_a, dmat_b)
            error_a = f_a @ dmat_a @ self.overlap - self.overlap @ dmat_a @ f_a
            error_b = f_b @ dmat_b @ self.overlap - self.overlap @ dmat_b @ f_b

            dmat_a, dmat_b = self.new_density_from_fock(f_a, f_b)
            current_e = 0.5 * (np.einsum('ij,ij', self.hcore, dmat_a + dmat_b) +
                               np.einsum('ij,ij', f_a, dmat_a) +
                               np.einsum('ij,ij', f_b, dmat_b))
            dE = abs(current_e - e_old)
            rmsd = np.linalg.norm(error_a + error_b)

            # print(error)
            print("{}\t{: 5.10f}\t{: 5.5f}\t{: 5.5f}".format(iter, current_e, dE,
                                                            rmsd))
            da_old = dmat_a.copy()
            db_old = dmat_b.copy()
            e_old = current_e
            iter += 1

        e_a, v_a = sp.linalg.eigh(f_a, b=self.overlap)
        e_b, v_b = sp.linalg.eigh(f_b, b=self.overlap)
        self.dmat = (dmat_a, dmat_b)
        self.fock = (f_a, f_b)
        self.mo_coeff = (v_a, v_b)
        self.mo_energies = (e_a, e_b)

    def solve_diis_fon(self):
        # is this standard to have two DIIS computations going simultaneously?
        diis = DIIS(self.diis_length)
        fon = FON()
        np.set_printoptions(linewidth=300)
        iter = 0
        dmat_a, dmat_b = self.core_density()
        da_old = dmat_a.copy()
        db_old = dmat_b.copy()
        e_old = np.trace(self.hcore @ (dmat_a + dmat_b))
        rmsd = 10
        print("Iter\tE(elec)\tdE\tRMSD\n")
        while iter < self.iter_max and rmsd > self.rmsd_eps:
            f_a, f_b = self.fock_mat(dmat_a, dmat_b)
            error_a = f_a @ dmat_a @ self.overlap - self.overlap @ dmat_a @ f_a
            error_b = f_b @ dmat_b @ self.overlap - self.overlap @ dmat_b @ f_b
            f_ab = diis.compute_new_vec(sp.linalg.block_diag(f_a, f_b),
                                        sp.linalg.block_diag(error_a, error_b))
            f_a, f_b = f_ab[:self.norbs, :self.norbs], f_ab[self.norbs:, self.norbs:]

            temperature = 10_000 * sum([np.linalg.norm(xx) for xx in diis.error_vecs])

            # dmat_a, dmat_b = self.new_density_from_fock(f_a, f_b)
            dmat_a, dmat_b = self.fon_density_from_fock(f_a, f_b, fon, temperature)

            current_e = 0.5 * (np.einsum('ij,ij', self.hcore, dmat_a + dmat_b) +
                               np.einsum('ij,ij', f_a, dmat_a) +
                               np.einsum('ij,ij', f_b, dmat_b))
            dE = abs(current_e - e_old)
            rmsd = np.linalg.norm(error_a + error_b)

            # print(error)
            print("{}\t{: 5.10f}\t{: 5.5f}\t{: 5.5f}\tTemp {: 5.5f}".format(iter, current_e, dE,
                                                             rmsd, temperature))
            da_old = dmat_a.copy()
            db_old = dmat_b.copy()
            e_old = current_e
            iter += 1

        e_a, v_a = sp.linalg.eigh(f_a, b=self.overlap)
        e_b, v_b = sp.linalg.eigh(f_b, b=self.overlap)
        self.dmat = (dmat_a, dmat_b)
        self.fock = (f_a, f_b)
        self.mo_coeff = (v_a, v_b)
        self.mo_energies = (e_a, e_b)


    def solve_diis(self):
        # is this standard to have two DIIS computations going simultaneously?
        diis = DIIS(self.diis_length)
        np.set_printoptions(linewidth=300)
        iter = 0
        dmat_a, dmat_b = self.core_density()
        da_old = dmat_a.copy()
        db_old = dmat_b.copy()
        e_old = np.trace(self.hcore @ (dmat_a + dmat_b))
        rmsd = 10
        print("Iter\tE(elec)\tdE\tRMSD\n")
        while iter < self.iter_max and rmsd > self.rmsd_eps:
            f_a, f_b = self.fock_mat(dmat_a, dmat_b)
            error_a = f_a @ dmat_a @ self.overlap - self.overlap @ dmat_a @ f_a
            error_b = f_b @ dmat_b @ self.overlap - self.overlap @ dmat_b @ f_b
            f_ab = diis.compute_new_vec(sp.linalg.block_diag(f_a, f_b),
                                        sp.linalg.block_diag(error_a, error_b))
            f_a, f_b = f_ab[:self.norbs, :self.norbs], f_ab[self.norbs:, self.norbs:]

            dmat_a, dmat_b = self.new_density_from_fock(f_a, f_b)
            current_e = 0.5 * (np.einsum('ij,ij', self.hcore, dmat_a + dmat_b) +
                               np.einsum('ij,ij', f_a, dmat_a) +
                               np.einsum('ij,ij', f_b, dmat_b))
            dE = abs(current_e - e_old)
            rmsd = np.linalg.norm(error_a + error_b)

            # print(error)
            print("{}\t{: 5.10f}\t{: 5.5f}\t{: 5.5f}".format(iter, current_e, dE,
                                                             rmsd))
            da_old = dmat_a.copy()
            db_old = dmat_b.copy()
            e_old = current_e
            iter += 1

        e_a, v_a = sp.linalg.eigh(f_a, b=self.overlap)
        e_b, v_b = sp.linalg.eigh(f_b, b=self.overlap)
        self.dmat = (dmat_a, dmat_b)
        self.fock = (f_a, f_b)
        self.mo_coeff = (v_a, v_b)
        self.mo_energies = (e_a, e_b)


if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    from pyscf import gto, scf
    import openfermion as of

    mol = gto.M(
        verbose=0,
        atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4',
        basis='sto-3g',
        charge=0,
        spin=None
    )
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    uhf = UHF(t + v, s, eri, 3, 2, iter_max=300,
              diis_length=4)

    uhf.solve_diis()
    print(uhf.dmat[0])
    print(uhf.dmat[1])

    uhf.solve_diis_fon()
    print(uhf.dmat[0])
    print(uhf.dmat[1])

    mf = scf.UHF(mol)
    mf.diis_space = 6
    mf.kernel()
    print(mf.energy_elec()[0])


