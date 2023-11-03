
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements
from qcpanop.pw_pbc.basis import plane_wave_basis
from qcpanop.pw_pbc.scf import uks

from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf

import ase
from ase.build import bulk
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import numpy as np

def print_gth_parameters(cell, basis):

    """

    print the GTH pseudopotential parameters

    :param cell: the unit cell
    :param basis: planewave basis information (contains gth parameters)

    """

    print('')
    print('    ==> GTH pseudopotential parameters (local) <==')
    print('')


    natom = len(cell._atom)

    for center in range (0, natom):

        params = basis.gth_params[center]

        print('        atom: %20s' % ( cell._atom[center][0] ) )
        print('        Zion: %20i' % ( params.Zion ) )
        print('        rloc: %20.8lf' % ( params.rloc ) )
        print('        c1:   %20.8lf' % ( params.local_cn[0] ) )
        print('        c2:   %20.8lf' % ( params.local_cn[1] ) )
        print('        c3:   %20.8lf' % ( params.local_cn[2] ) )
        print('        c4:   %20.8lf' % ( params.local_cn[3] ) )
        print('')

    print('    ==> GTH pseudopotential parameters (non-local) <==')
    print('')

    for center in range (0, natom):

        params = basis.gth_params[center]

        print('        atom: %20s' % ( cell._atom[center][0] ) )
        rl = params.rl
        print('        rl:   %20.8lf %20.8lf %20.8lf' % ( rl[0], rl[1], rl[2] ) )
        print('        h^l_ij:')

        for l in range (0, 3):
            h = params.hgth
            print('')
            print('            l = %i' % ( l ) )
            for i in range (0, 3):
                print('            %20.8lf %20.8lf %20.8lf' % ( h[l, i, 0], h[l, i, 1], h[l, i, 2] ) )
        print('')

def main():

    # define unit cell 
    
    a = np.eye(3) * 4.0
    atom = 'Mn 0 0 0'

    #ase_atom = bulk('Si', 'diamond', a = 10.26)
    #ase_atom = bulk('C', 'diamond', a = 6.74)
    #ase_atom = bulk('H', 'diamond', a = 8.88)
    #ase_atom = bulk('Ne', 'diamond', a = 10.26)

    #atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    #a = ase_atom.cell 
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'cc-pvdz',
                 pseudo = 'GTH-LDA-q7',
                 #verbose = 100,
                 #ke_cutoff = 500 / 27.21138602,
                 precision = 1.0e-8,
                 #charge = 0,
                 spin = 1,
                 dimension = 3)
    
    cell.build()

    cutoff = 500.0

    # get plane wave basis information
    basis_full = plane_wave_basis(cell, 
        ke_cutoff = cutoff / 27.21138602, 
        n_kpts = [1, 1, 1],
        approximate_nl_pp = False)

    basis_approx = plane_wave_basis(cell, 
        ke_cutoff = cutoff / 27.21138602, 
        n_kpts = [1, 1, 1],
        approximate_nl_pp = True)

    # run plane wave scf 
    en_full = uks(cell, basis_full, xc = 'lda', guess_mix = True)
    en_approx = uks(cell, basis_approx, xc = 'lda', guess_mix = True)

    print("    nbf, en(full), en(approx), diff: %8i %20.12lf %20.12lf %20.12lf" % (basis_full.n_plane_waves_per_k[0], en_full, en_approx, en_approx - en_full))

if __name__ == "__main__":
    main()
