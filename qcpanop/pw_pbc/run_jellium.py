
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements
from qcpanop.pw_pbc.basis import plane_wave_basis 
from qcpanop.pw_pbc.scf import uks

from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf

import ase
from ase.build import bulk
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import numpy as np

def main():

    # define unit cell 
   
    # omega = 2^18 a0^3 
    a = np.eye(3) * 64.0
    atom = 'He 0 0 0'
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'sto-3g', #'cc-pvdz',
                 pseudo = 'gth-pbe',
                 #verbose = 100,
                 ke_cutoff = 0.01 / 27.21138602,
                 precision = 1.0,
                 charge = 0,
                 spin = 0,
                 dimension = 3)

    cell.build()

    cutoff = 3.22
    #cutoff = 12.917

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
        ke_cutoff = cutoff / 27.21138602, 
        n_kpts = [1, 1, 1])

    # run plane wave scf 
    en, ca, cb = uks(cell, basis, xc = 'hf', guess_mix = False, maxiter = 1, ace_exchange = True, jellium = True, jellium_ne = 100)

    # jellium / 3.22 ev / omega = 2^18 / ne = 100
    assert np.isclose(en, -3.932033059623)

    # C / diamond / hf / gth-pbe / 500 ev cutoff
    #assert np.isclose(en, -10.061105991782)

    # C / diamond / hf / gth-pbe / 1000 ev cutoff
    #assert np.isclose(en, -10.236907018105)

    # C / diamond / hf / gth-pbe / 2000 ev cutoff
    #assert np.isclose(en, -10.249995429892)

    # C / diamond / hf / gth-pbe / 3000 ev cutoff
    #assert np.isclose(en, -10.25032156584)


if __name__ == "__main__":
    main()
