"""
Implementation of basic CCD equations
like psi4numpy but through OpenFermion
"""
from itertools import product

import numpy as np
from numpy import einsum

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial



def get_h2o():
    from openfermionpsi4 import run_psi4
    geometry = [['O', [0.000000000000, 0.000000000000, -0.068516219320]],
                ['H', [0.000000000000, -0.790689573744, 0.543701060715]],
                ['H', [0.000000000000, 0.790689573744, 0.543701060715]]]
    multiplicity = 1
    charge = 0
    basis = 'sto-3g'
    molecule = of.MolecularData(geometry=geometry, multiplicity=multiplicity,
                                charge=charge, basis=basis)
    molecule = run_psi4(molecule)
    print(molecule.hf_energy)
    return molecule



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


class CCD:

    def __init__(self, oei=None, tei=None, molecule=None, iter_max=100, e_convergence=1.0E-6,
                 r_convergence=1.0E-6, nalpha=None, nbeta=None, escf=None, orb_energies=None):
        self.molecule = molecule

        self.t0 = None
        self.sigma = None
        self.iter_max = iter_max
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence


        if molecule is not None:
            # if molecule is defined
            if self.molecule.multiplicity != 1:
                raise ValueError("We are only implementing for closed shell RHF")

            self.n_alpha = molecule.n_electrons // 2  # sz = 0 apparently
            self.n_beta = molecule.n_electrons // 2

            # self.spin_orbs = np.kron(molecule.canonical_orbitals, np.eye(2))
            oei, tei = molecule.get_integrals()
            soei, stei = spinorb_from_spatial(oei, tei)
            self.astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
            self.soei = soei


            self.orb_e = molecule.orbital_energies
            self.sorb_e = np.vstack((self.orb_e, self.orb_e)).flatten(order='F')
            self.scf_energy = molecule.hf_energy

        else:
            if not all(v is not None for v in
                       [oei, tei, nalpha, nbeta, escf, orb_energies]):
                raise ValueError("Not all inputs specified")
            self.n_alpha = nalpha
            self.n_beta = nbeta

            self.astei = tei
            self.soei = oei
            self.sorb_e = orb_energies
            self.scf_energy = escf

        self.gstei = self.astei.transpose(0, 1, 3, 2)
        self.norbs = self.astei.shape[0]//2
        self.nso = 2 * self.norbs

        self.nocc = self.n_alpha + self.n_beta
        self.nvirt = self.nso - self.nocc

        self.n = np.newaxis
        self.o = slice(None, self.nocc)
        self.v = slice(self.nocc, None)
        n, o, v = self.n, self.o, self.v
        self.e_abij = 1 / (-self.sorb_e[v, n, n, n] - self.sorb_e[n, v, n, n] +
                           self.sorb_e[n, n, o, n] + self.sorb_e[n, n, n, o])

        self.t_amp = np.zeros((self.nvirt, self.nvirt, self.nocc, self.nocc))

    def solve_for_amplitudes(self):
        t2, h, g = self.t_amp, self.soei, self.gstei
        o, v = self.o, self.v
        e_abij = self.e_abij
        old_energy = ccd_energy(t2, h, g, o, v)
        for idx in range(self.iter_max):

            doubles_res = doubles_residual_contractions(t2, g, o, v)

            new_doubles = doubles_res * e_abij

            current_energy = ccd_energy(new_doubles, h, g, o, v)
            delta_e = np.abs(old_energy - current_energy)

            if delta_e < self.e_convergence:
                self.t_amp = new_doubles
                self.ccd_energy = current_energy
                return new_doubles
            else:
                t2 = new_doubles
                old_energy = current_energy
                print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx,
                                                                        old_energy,
                                                                        delta_e))
        else:
            print("Did not converge")
            self.t_amp = new_doubles
            self.ccd_energy = current_energy
            return new_doubles

    def compute_energy(self, t_amplitudes):
        o, v = self.o, self.v
        gmo = self.astei
        return (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v], t_amplitudes, optimize=True)

    def compute_energy_ncr(self):
        """Amplitude equations determined from pdaggerq"""
        pass

    def _amplitude_zero_nonpairs(self, tamps):
        """Zero out non-pair components of the amplitudes

        the pCCD ansatz is sum_{ia}t_{i}^{a}P_{a}^ P_{i} where
        P_{x}^ = a_{x alpha}^ a_{x beta}^

        :param tamps: (virt, virt, occ, occ) corresponding to t^{ab}_{ij} in
                      spin-orbitals {a,b,i,j}
        """
        pccd_t2_amps = np.zeros_like(tamps)
        amp_count = 0
        for a, b in product(range(self.nvirt), repeat=2):
            for i, j in product(range(self.nocc), repeat=2):
                a_spatial, b_spatial = a // 2, b // 2
                i_spatial, j_spatial = i // 2, j // 2
                if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and j % 2 == 1 and a_spatial == b_spatial and i_spatial == j_spatial:
                    pccd_t2_amps[a, b, i, j] = tamps[a, b, i, j]
                    pccd_t2_amps[b, a, j, i] = tamps[a, b, i, j]

                    pccd_t2_amps[a, b, j, i] = -tamps[a, b, i, j]
                    pccd_t2_amps[b, a, i, j] = -tamps[a, b, i, j]

                    amp_count += 1
        return pccd_t2_amps

    def get_pccd_amps(self, tamps):
        """
            for i in range(o):
                for a in range(v):
                    self.residual[i * v + a]
        :param tamps:
        :return:
        """
        pccd_t2_amps = np.zeros((self.nvirt//2, self.nocc//2))
        amp_count = 0
        for a, b in product(range(self.nvirt), repeat=2):
            for i, j in product(range(self.nocc), repeat=2):
                a_spatial, b_spatial = a // 2, b // 2
                i_spatial, j_spatial = i // 2, j // 2
                if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and j % 2 == 1 and a_spatial == b_spatial and i_spatial == j_spatial:
                    pccd_t2_amps[a_spatial, i_spatial] = tamps[a, b, i, j]
                    amp_count += 1
        assert np.isclose((self.nvirt//2) * (self.nocc//2), amp_count)
        return pccd_t2_amps

    def pccd_solve(self, starting_amps=None):
        if starting_amps is None:
            t_amp = self.t_amp
        else:
            t_amp = starting_amps

        gmo = self.astei
        n, o, v = self.n, self.o, self.v
        e_abij = self.e_abij

        # Initialize energy
        E_CCD = 0.0

        for cc_iter in range(1, self.iter_max + 1):
            E_old = E_CCD

            t_amp = self._amplitude_zero_nonpairs(t_amp)

            # Collect terms
            mp2 = gmo[v, v, o, o]

            cepa1 = (1 / 2) * np.einsum('abcd, cdij -> abij', gmo[v, v, v, v],
                                        t_amp, optimize=True)
            cepa2 = (1 / 2) * np.einsum('klij, abkl -> abij', gmo[o, o, o, o],
                                        t_amp, optimize=True)
            cepa3a = np.einsum('akic, bcjk -> abij', gmo[v, o, o, v], t_amp,
                               optimize=True)
            cepa3b = -cepa3a.transpose(1, 0, 2, 3)
            cepa3c = -cepa3a.transpose(0, 1, 3, 2)
            cepa3d = cepa3a.transpose(1, 0, 3, 2)
            cepa3 = cepa3a + cepa3b + cepa3c + cepa3d

            ccd1a_tmp = np.einsum('klcd,bdkl->cb', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd1a = np.einsum("cb,acij->abij", ccd1a_tmp, t_amp, optimize=True)

            ccd1b = -ccd1a.transpose(1, 0, 2, 3)
            ccd1 = -(1 / 2) * (ccd1a + ccd1b)

            ccd2a_tmp = np.einsum('klcd,cdjl->jk', gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd2a = np.einsum("jk,abik->abij", ccd2a_tmp, t_amp, optimize=True)

            ccd2b = -ccd2a.transpose(0, 1, 3, 2)
            ccd2 = -(1 / 2) * (ccd2a + ccd2b)


            ccd3_tmp = np.einsum("klcd,cdij->klij", gmo[o, o, v, v], t_amp,
                                 optimize=True)
            ccd3 = (1 / 4) * np.einsum("klij,abkl->abij", ccd3_tmp, t_amp,
                                       optimize=True)

            ccd4a_tmp = np.einsum("klcd,acik->laid", gmo[o, o, v, v], t_amp,
                                  optimize=True)
            ccd4a = np.einsum("laid,bdjl->abij", ccd4a_tmp, t_amp,
                              optimize=True)

            ccd4b = -ccd4a.transpose(0, 1, 3, 2)
            ccd4 = (ccd4a + ccd4b)

            # Update Amplitude
            residual = self._amplitude_zero_nonpairs(mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)
            t_amp_new = -e_abij * residual
            t_amp_new = self._amplitude_zero_nonpairs(t_amp_new)

            # Evaluate Energy
            E_CCD = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v],
                                        -t_amp_new, optimize=True)
            t_amp = t_amp_new
            dE = E_CCD - E_old
            print('pCCD Iteration %3d: Energy = %4.12f dE = %1.5E' % (
            cc_iter, E_CCD, dE))

            if abs(dE) < self.e_convergence:
                print("\npCCD Iterations have converged!")
                break

        print('\npCCD Correlation Energy:    %15.12f' % (E_CCD))
        print('pCCD Total Energy:         %15.12f' % (E_CCD + self.scf_energy))
        self.t_amp = t_amp


if __name__ == "__main__":
    import copy
    np.set_printoptions(linewidth=500)
    molecule = get_h2o()
    ccd = CCD(molecule=molecule, iter_max=20)
    ccd.solve_for_amplitudes()
    exit()

    ccd.t_amp = ccd._amplitude_zero_nonpairs(ccd.t_amp)
    print("Correlation energy from just pCCD amps ", ccd.compute_energy(ccd.t_amp))
    print("Correlation energy from just pCCD amps ", ccd.compute_energy(ccd.t_amp) + ccd.molecule.hf_energy)

    pccd = CCD(molecule=molecule, iter_max=20)
    pccd.pccd_solve()

    pccd_amps = pccd.get_pccd_amps(pccd.t_amp)
    t2_test = np.zeros((pccd.nocc//2) * (pccd.nvirt//2))
    for i in range(pccd.nocc//2):
        for a in range(pccd.nvirt//2):
            t2_test[i * (pccd.nvirt//2) + a] = pccd_amps[a, i]
    print(t2_test.shape)

