"""
Lattice model mean-field.

The SCF equations for lattice equations

we do this in the full antisymmetric space
"""
import numpy as np
import scipy as sp


class LatticeSCF:

    def __init__(self, oei, tei, nalpha, nbeta, type='rhf',
                 iter_max=50, rmsd_eps=1.0E-8, diis_length=6):
        """

        :param oei: spin-orbital 1-electron integrals
        :param tei: spin-orbital 2-electron integrals
        :param nalpha: number of alpha electrons
        :param nbeta: number of beta electrons
        :param type: scf-type rhf, uhf, ghf
        :param iter_max: maximum scf iterations
        :param rmsd_eps:
        :param diis_length:
        """
        self.type = type
        self.oei = oei
        self.tei = tei
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.nso = oei.shape[0]

        self.occa = [2 * x for x in range(self.nalpha)]
        self.occb = [2 * x + 1 for x in range(self.nbeta)]

        self.iter_max = iter_max
        self.rmsd_eps = rmsd_eps
        self.diis_length = diis_length

    def core_density(self):
        _, c = sp.linalg.eigh(self.oei)
        rho_a = c[:, self.occa] @ c[:, self.occa].T
        rho_b = c[:, self.occb] @ c[:, self.occb].T
        return rho_a, rho_b

    def j_mat(self, rho_t):
        return np.einsum('ijkl,jk->il', self.tei, rho_t)
        # <1'2'|21>

    def k_mat(self, rho_spin):
        return np.einsum('ijlk,jk->il', self.tei, rho_spin)

    def fock_mat(self, rho_a, rho_b):
        f = self.oei + self.j_mat(rho_a + rho_b)
        f -= self.k_mat(rho_a + rho_b)
        return f

    def new_density_from_fock(self, fock):
        _, c = sp.linalg.eigh(fock)
        rho_a = c[:, self.occa] @ c[:, self.occa].T
        rho_b = c[:, self.occb] @ c[:, self.occb].T
        return rho_a, rho_b


    def solve_scf(self):
        rho_a, rho_b = self.core_density()
        f = self.fock_mat(rho_a, rho_b)
        print(f)
        print((rho_a + rho_b) @ f - f @ (rho_a + rho_b))

        print(np.einsum('ij,ij', rho_a + rho_b, f))
        exit()
        rho_a, rho_b = self.new_density_from_fock(f)

        f = self.fock_mat(rho_a, rho_b)
        rho_a, rho_b = self.new_density_from_fock(f)

        f = self.fock_mat(rho_a, rho_b)
        rho_a, rho_b = self.new_density_from_fock(f)


if __name__ == "__main__":
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from itertools import product
    dim = 6
    sdim = 2 * dim
    bcs_oei = np.arange(1, dim + 1)
    bcs_oei = np.diag(np.kron(bcs_oei, np.ones(2)))
    bcs_stei = np.zeros((sdim, sdim, sdim, sdim))
    bcs_coupling = 1
    for p, q in product(range(dim), repeat=2):
        bcs_stei[2 * p, 2 * q + 1, 2 * q + 1, 2 * p] = -bcs_coupling

    import openfermion as of

    # bcs_ham = of.InteractionOperator(0, bcs_oei, bcs_stei)
    # print(of.get_fermion_operator(bcs_ham))
    # bcs_ham_sparse = of.get_sparse_operator(of.get_fermion_operator(bcs_ham))
    # w, v = np.linalg.eigh(bcs_ham_sparse.toarray())
    # print(w[:10])

    mf = LatticeSCF(oei=bcs_oei, tei=bcs_stei, nalpha=dim//2, nbeta=dim//2)
    mf.solve_scf()