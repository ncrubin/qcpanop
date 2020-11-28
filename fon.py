"""
Fractional occupation number (FON) utilities for SCF convergence

Implementation of FON-HF technique.
JCP 110 2, 1999
"""
import numpy as np

# set up constants
BOLTZMANN = 1.38064852E-23  # J / K
j_to_h = 2.293710449E17
BOLTZMANN_H = BOLTZMANN * j_to_h

class FON:

    def __init__(self):
        pass

    def broadening_param_to_temp(self, beta):
        return 1 / (BOLTZMANN_H * beta)

    def get_start_temp(self, dmat, fock, overlap):
        error = fock @ dmat @ overlap - overlap @ dmat @ fock
        return 10_000 * np.linalg.norm(error)

    def pfon_rhf(self, temp: float, orb_energies: np.ndarray, num_alpha):
        """pseudo-fractional-occupation.  Fermi energy is defined as
        e_HOMO + e_LUMO / 2.  Occupations do not necessarily sum to
        the target number of electrons so we need to normalize the occs.

        Effectively this means that for a given temperature a higher
        number of virtuals are fractionally occupied.
        """
        e_homo = orb_energies[num_alpha - 1]
        e_lumo = orb_energies[num_alpha]
        e_fermi = (e_lumo + e_homo) / 2
        n_i = np.array(self.fermi_dirac(orb_energies, e_fermi, temp))
        return n_i * num_alpha / np.sum(n_i)

    def frac_occ_rhf(self, temp: float, orb_energies: np.ndarray, num_alpha):
        """
        Determine fractional occupations for a given temperature

        The Fermi energy is determined by bisection such thtat

        \sum_{i}n_{i} = N

        where n_{i} is

        n_{i} = 1 / (1 + e^{beta(e_{i} - e_{f})})

        and N is the number of electrons

        For the binary search if e_f = e_{0} then occupation is 0.5
        if e_f = e_{-1}  then occupation is (norb-1) + 0.5
        """
        e_fermi = self.bisection_fermi_energy_search(orb_energies[0],
                                                     orb_energies[-1],
                                                     num_alpha,
                                                     orb_energies,
                                                     temp
                                                     )
        return self.fermi_dirac(orb_energies, e_fermi, temp)

    def bisection_fermi_energy_search(self, e_f_init_low: float,
                                      e_f_init_high: float, target_n: int,
                                      orb_e: np.ndarray, temp: float) -> float:
        """
        Bisection search for the fermi energy
        """
        current_score = 0
        low_val = e_f_init_low
        high_val = e_f_init_high
        while not np.isclose(current_score, target_n):
            middle_val = (low_val + high_val) / 2
            middle_score = sum(self.fermi_dirac(orb_e, middle_val, temp))
            if middle_score < target_n:
                low_val = middle_val
            else:
                high_val = middle_val
            current_score = middle_score
        return middle_val

    def fermi_dirac(self, e_i: np.ndarray, e_f: float, temp: float):
        """save fermi-dirac avoids numerical overflow"""
        beta = 1 / (BOLTZMANN_H * temp)
        exp_val = beta * (e_i - e_f)
        # print("x ", exp_val)
        # print("1 + e^x", 1 + np.exp(exp_val))
        # print("1 + e^-x", 1 + np.exp(-exp_val))
        safe_fd = []
        for xx in exp_val:
            if xx > 10:
                sfd = np.exp(-xx) / (1 + np.exp(-xx))
            elif xx < -10:
                sfd = 1 / (1 + np.exp(xx))
            else:
                sfd = 1 / (1 + np.exp(xx))
            safe_fd.append(sfd)
        return safe_fd


if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    from pyscf import gto, scf
    import openfermion as of
    from rhf import RHF

    # mol = gto.M(
    #     verbose=0,
    #     atom='O   0.000000000000  -0.143225816552   0.000000000000;H  1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000',
    #     basis='sto-3g',
    # )
    mol = gto.M(
        verbose=0,
        atom='Li 0 0 0; H 0 0 5.0',
        basis='sto-3g',
    )
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    rhf = RHF(t + v, s, eri, mol.nelectron, iter_max=3,
              diis_length=4)
    rhf.solve_diis()
    fon = FON()
    mo_e = rhf.mo_energies
    # print(mo_e)
    # print(sum(fon.fermi_dirac(rhf.mo_energies, rhf.mo_energies[0], 10)))
    # print(sum(fon.fermi_dirac(rhf.mo_energies, rhf.mo_energies[-1], 10)))
    # print()
    # e_f = fon.bisection_fermi_energy_search(mo_e[0], mo_e[-1], rhf.nelec // 2,
    #                                         mo_e, 10)
    # print(fon.fermi_dirac(rhf.mo_energies, e_f, 10))
    # n_i = fon.frac_occ_rhf(500000, mo_e, 2)
    n_i = fon.pfon_rhf(50000, mo_e, 2)
    print(n_i)
    print(sum(n_i))