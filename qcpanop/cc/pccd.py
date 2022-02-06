"""
C-like implementaiton of of pCCD equations.

TODO: vectorize further with numpy.  current form is to compare with other codes
"""
from itertools import product
import numpy as np
import openfermion as of


def get_h2o():
    # from openfermionpsi4 import run_psi4
    from openfermionpyscf import run_pyscf
    geometry = [['O', [0.000000000000, 0.000000000000, -0.068516219320]],
                ['H', [0.000000000000, -0.790689573744, 0.543701060715]],
                ['H', [0.000000000000, 0.790689573744, 0.543701060715]]]
    multiplicity = 1
    charge = 0
    basis = '6-31g'
    molecule = of.MolecularData(geometry=geometry, multiplicity=multiplicity,
                                charge=charge, basis=basis)
    molecule = run_pyscf(molecule)
    print(molecule.hf_energy)
    return molecule


class pCCD:

    def __init__(self, molecule=None, iter_max=100, e_convergence=1.0E-6,
                 r_convergence=1.0E-6, oei=None, tei=None, n_electrons=None,
                 enuc=0):
        self.molecule = molecule
        self.t0 = None
        self.sigma = None
        self.iter_max = iter_max
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence

        if molecule is None and (oei is not None and tei is not None and n_electrons is not None):
            self.oei = oei
            self.tei = tei
            self.o = n_electrons // 2
            norbs = oei.shape[0]
            self.v = norbs - self.o
            self.enuc = enuc
        else:
            self.o = molecule.n_electrons // 2
            self.v = molecule.n_orbitals - self.o
            oei, tei = molecule.get_integrals()
            self.oei = oei
            self.tei = tei
            self.enuc = molecule.nuclear_repulsion

    def setup_integrals(self):
        oei, tei = self.oei, self.tei
        o, v = self.o, self.v

        self.v_iiaa = np.zeros(o * v)
        self.v_iaia = np.zeros(o * v)
        self.v_ijij = np.zeros(o * o)
        self.v_abab = np.zeros(v * v)
        self.f_o = np.zeros(o)
        self.f_v = np.zeros(v)

        # print("v_(ii|aa)")
        for i in range(o):
            for a in range(v):
                self.v_iiaa[i * v + a] = tei[i, a + o, a + o, i]

        # print("v_(ia|ia)")
        for i in range(o):
            for a in range(v):
                self.v_iaia[i * v + a] = tei[i, i, a + o, a + o]

        # print("v_(ij|ij)")
        for i in range(o):
            for j in range(o):
                self.v_ijij[i * o + j] = tei[i, i, j, j]

        # print("v_(ab|ab)")
        for a in range(v):
            for b in range(v):
                self.v_abab[a * v + b] = tei[a + o,  a + o, b + o, b + o]

        # print("fock (o)")
        for i in range(o):
            dum = oei[i, i]
            for k in range(o):
                dum += 2 * tei[i, k, k, i]
                dum -= tei[i, k, i, k]
            self.f_o[i] = dum

        # print("fock (v)")
        for a in range(v):
            dum = oei[a + o,  a + o]
            for k in range(o):
                dum += 2 * tei[a + o, k, k, a + o]
                dum -= tei[a + o, k, a + o, k]
            self.f_v[a] = dum

        self.escf = self.enuc # molecule.nuclear_repulsion
        for i in range(o):
            self.escf += oei[i, i] + self.f_o[i]

    def compute_energy(self):
        o, v = self.o, self.v
        en = 0

        # initialize amplitudes to zero
        self.t2 = np.zeros(o * v)
        self.setup_integrals()
        iter = 0

        while iter < self.iter_max:

            self.evaluate_residual()

            # update amplitudes
            for i in range(o):
                for a in range(v):
                    self.residual[i * v + a] *= -0.5 / (self.f_v[a] - self.f_o[i])

            self.t0 = self.residual.copy()
            self.t0 = self.t0 + -self.t2

            nrm = np.linalg.norm(self.t0)
            self.t2 = self.residual.copy()

            dE = en
            en = self.evaluate_projected_energy()
            dE -= en
            print("\t\t\t{}\t{: 5.10f}\t{: 5.10f}\t{: 5.10f}".format(iter, en, dE, nrm))

            if np.abs(dE) < self.e_convergence and nrm < self.r_convergence:
                break

            iter += 1

        self.correlation_energy = en
        self.total_energy = self.escf + en
        print("\t\tIterations Converged")
        print("\t\tCorrelation Energy {: 5.20f}".format(self.total_energy - self.escf))
        print("\t\tTotal Energy {: 5.20f}".format(self.total_energy))


    def evaluate_projected_energy(self):
        o, v = self.o, self.v
        energy = 0.
        # reset t0 to ones
        for i in range(o):
            for a in range(v):
                energy += self.t2[i * v + a] * self.v_iaia[i * v + a]
        return energy

    def normalize(self):
        self.t0 = np.ones(self.o * self.v)

    def evaluate_residual(self):
        o, v = self.o, self.v
        self.normalize()
        self.residual = np.zeros(o * v)
        self.residual = self.evaluate_sigma()

        VxT_v = np.zeros(v)
        VxT_o = np.zeros(o)
        VxT_oo = np.zeros(o * o)

        # print("VxT_v ")
        for a in range(v):
            # contract over the occupied space to get the virtual index
            VxT_v[a] = -2.0 * np.dot(self.v_iaia[a::v], self.t2[a::v])

        # print("VxT_o ")
        for i in range(o):
            # contract over the virtual index of the vectorized matrix
            VxT_o[i] = -2.0 * np.dot(self.v_iaia[i * v:(i + 1) * v], self.t2[i * v: (i + 1)* v ])

        # print("VxT_oo(i,j) = (jb|jb) t(i,b)")
        for i, j in product(range(o), repeat=2):
            VxT_oo[i * o + j] = np.dot(self.v_iaia[j*v:(j + 1)*v], self.t2[i*v:(i + 1)*v])

        # // r2(i,a) += t(j,a) VxT_oo(i,j)
        for i in range(o):
            for a in range(v):
                # sum over j index
                self.residual[i * v + a] += np.dot(self.t2[a::v], VxT_oo[i * o:(i + 1) * o])

        # print("VxT_v and o contraction")
        for i in range(o):
            for a in range(v):
                dum = 0.
                dum += VxT_v[a] * self.t2[i * v + a]
                dum += VxT_o[i] * self.t2[i * v + a]

                t_t2 = self.t2[i * v + a]
                dum += 2.0 * self.v_iaia[i * v + a] * t_t2 * t_t2
                self.residual[i * v + a] += dum

    def evaluate_sigma(self):
        """
        Evaluate
        """
        o, v = self.o, self.v
        sigma = self.v_iaia.copy()
        for i in range(o):
            for a in range(v):
                sigma[i * v + a] -= 2 * (2 * self.v_iiaa[i * v + a] - self.v_iaia[i * v + a]) * self.t2[i * v + a]

        for i in range(o):
            for a in range(v):
                dum = 0
                for b in range(v):
                    dum += self.v_abab[a * v + b] * self.t2[i * v + b]
                for j in range(o):
                    dum += self.v_ijij[i * o + j] * self.t2[j * v + a]
                sigma[i * v + a] += dum
        return sigma


if __name__ == "__main__":
    molecule = get_h2o()
    pccd = pCCD(molecule, iter_max=20)
    pccd.setup_integrals()
    pccd.compute_energy()

    print("pCCD T2 amps")
    for i in range(pccd.o):
        for a in range(pccd.v):
            print("{}\t{}\t{: 5.20f}".format(i, a, pccd.t2[i * pccd.v + a]))

    from lambda_ccd_with_pccd_restriction import LambdaCCD
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from lambda_ccd import kernel, ccsd_energy
    # pccd_lccd = LambdaCCD(molecule, restrict_to_pair_doubles=False)
    # oei, tei = molecule.oei, pccd.tei
    oei, tei = molecule.get_integrals()
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    gtei = astei.transpose(0, 1, 3, 2)

    mf = molecule._pyscf_data['scf']
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    norbs = oei.shape[0]
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)



    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', gtei[o, o, o, o])
    print("HF energies")
    print(hf_energy, mf.e_tot - mf.energy_nuc())


    # pccd_lccd.solve_cc_equations(soei, astei)
    t1z, t2z = np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1f, t2f, l1f, l2f = kernel(t1z, t2z, fock, gtei, o, v, e_ai, e_abij,
                                stopping_eps=1.0E-12)

    print("Final Correlation Energy")
    print(ccsd_energy(t1f, t2f, fock, gtei, o, v) - hf_energy)

    print("E(CCSD)-NCR = {}".format(ccsd_energy(t1f, t2f, fock, gtei, o, v) + mf.energy_nuc()))

    mycc = mf.CCSD()
    mycc.conv_tol = 1.0E-12
    ecc, pyscf_t1, pyscf_t2 = mycc.kernel()
    print('CCSD correlation energy', mycc.e_corr)



    lccd = LambdaCCD(molecule=molecule, e_convergence=mycc.conv_tol)
    lccd.solve_cc_equations(soei, astei)


    from pyscf import cc
    mycc = cc.CCSD(mf)
    old_update_amps = mycc.update_amps
    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return t1 * 0, t2
    mycc.update_amps = update_amps
    mycc.kernel()

    lccd = LambdaCCD(molecule=molecule, e_convergence=mycc.conv_tol, restrict_to_pair_doubles=True,
                     iter_max=500)
    lccd.solve_cc_equations(soei, astei)

    pccd = pCCD(molecule, iter_max=20)
    pccd.setup_integrals()
    pccd.compute_energy()



