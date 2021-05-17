"""
Implementation of basic CCD equations
like psi4numpy but through OpenFermion
"""
from itertools import product
import numpy as np
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


class CCD:

    def __init__(self, molecule, iter_max=100, e_convergence=1.0E-6,
                 r_convergence=1.0E-6):
        self.molecule = molecule

        self.t0 = None
        self.sigma = None
        self.iter_max = iter_max
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence
        self.norbs = molecule.n_orbitals
        self.nso = 2 * self.norbs

        if self.molecule.multiplicity != 1:
            raise ValueError("We are only implementing for closed shell RHF")

        self.n_alpha = molecule.n_electrons // 2  # sz = 0 apparently
        self.n_beta = molecule.n_electrons // 2

        self.nocc = self.n_alpha + self.n_beta
        self.nvirt = self.nso - self.nocc

        self.spin_orbs = np.kron(molecule.canonical_orbitals, np.eye(2))
        oei, tei = molecule.get_integrals()
        soei, stei = spinorb_from_spatial(oei, tei)
        self.astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

        self.orb_e = molecule.orbital_energies
        self.sorb_e = np.vstack((self.orb_e, self.orb_e)).flatten(order='F')

        self.n = np.newaxis
        self.o = slice(None, self.nocc)
        self.v = slice(self.nocc, None)
        n, o, v = self.n, self.o, self.v
        self.e_abij = 1 / (-self.sorb_e[v, n, n, n] - self.sorb_e[n, v, n, n] +
                           self.sorb_e[n, n, o, n] + self.sorb_e[n, n, n, o])

        self.t_amp = np.zeros((self.nvirt, self.nvirt, self.nocc, self.nocc))

    def compute_energy(self):
        """
        Compute the CCD amplitudes

        Iteration code taken from https://github.com/psi4/psi4numpy/blob/master/Tutorials/08_CEPA0_and_CCD/8b_CEPA0_and_CCD.ipynb
        """
        t_amp = self.t_amp
        gmo = self.astei
        n, o, v = self.n, self.o, self.v
        e_abij = self.e_abij

        # Initialize energy
        E_CCD = 0.0

        for cc_iter in range(1, self.iter_max + 1):
            E_old = E_CCD

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
            t_amp_new = e_abij * (
                        mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)

            # Evaluate Energy
            E_CCD = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v],
                                        t_amp_new, optimize=True)
            t_amp = t_amp_new
            dE = E_CCD - E_old
            print('CCD Iteration %3d: Energy = %4.12f dE = %1.5E' % (
            cc_iter, E_CCD, dE))

            if abs(dE) < 1.e-8:
                print("\nCCD Iterations have converged!")
                break

        print('\nCCD Correlation Energy:    %15.12f' % (E_CCD))
        print('CCD Total Energy:         %15.12f' % (E_CCD + molecule.hf_energy))






if __name__ == "__main__":
    np.set_printoptions(linewidth=500)
    molecule = get_h2o()
    ccd = CCD(molecule, iter_max=20)
    ccd.compute_energy()


