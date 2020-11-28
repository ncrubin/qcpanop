"""
Define the UNO orbitals as a prep-processing
step
"""
from uhf import UHF
import scipy as sp


def uno(uhf: UHF):
    d_uhf = sum(uhf.dmat)
    s = uhf.overlap
    ws, vs = np.linalg.eigh(s)
    assert np.allclose(vs @ np.diag(ws) @ vs.T, s)
    snhalf = vs @ np.diag(ws**-0.5) @ vs.T
    shalf = vs @ np.diag(ws**0.5) @ vs.T
    # sigma, c = sp.linalg.eigh(s @ d_uhf @ s, b=s)
    sigma, c = np.linalg.eigh(shalf @ d_uhf @ shalf)
    c = snhalf @ c
    print(sigma)




if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(linewidth=300)
    from pyscf import gto, scf
    import openfermion as of
    # mol = gto.M(
    #     verbose=0,
    #     atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4',
    #     basis='sto-3g',
    #     charge=0,
    #     spin=None
    # )
    mol = gto.M(
        verbose=0,
        atom='Cr 0 0 0; Cr 0 0 3.4',
        basis='6-31g*',
        charge=0,
        spin=None
    )
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    uhf = UHF(t + v, s, eri, 3, 2, iter_max=300,
              diis_length=4)
    uhf.solve_diis_fon()

    uno(uhf)