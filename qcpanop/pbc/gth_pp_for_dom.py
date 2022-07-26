import numpy as np

from pw_scf import plane_wave_basis
from pw_scf import get_gth_pseudopotential

from pyscf.pbc import gto

def print_gth_parameters(cell, basis):

    """

    print the GTH pseudopotential parameters

    :param cell: the unit cell
    :param basis: planewave basis information (contains gth parameters)

    """

    print('')
    print('    ==> GTH pseudopotential parameters (local) <==')
    print('')

    params = basis.gth_params

    natom = len(cell._atom)

    for center in range (0, natom):
        print('        atom: %20s' % ( cell._atom[center][0] ) )
        print('        Zion: %20i' % ( params.Zion[center] ) )
        print('        rloc: %20.12lf' % ( params.rloc[center] ) )
        print('        c1:   %20.12lf' % ( params.local_cn[center][0] ) )
        print('        c2:   %20.12lf' % ( params.local_cn[center][1] ) )
        print('        c3:   %20.12lf' % ( params.local_cn[center][2] ) )
        print('        c4:   %20.12lf' % ( params.local_cn[center][3] ) )
        print('')

    print('    ==> GTH pseudopotential parameters (non-local) <==')
    print('')

    for center in range (0, natom):

        print('        atom: %20s' % ( cell._atom[center][0] ) )
        print('        rl:   %20.12lf %20.12lf %20.12lf' % ( params.rl[center][0], params.rl[center][1], params.rl[center][2] ) )
        print('        h^l_ij:')

        for l in range (0, 3):
            print('')
            print('            l = %i' % ( l ) )
            for i in range (0, 3):
                print('            %20.12lf %20.12lf %20.12lf' % ( params.hgth[center, l, i, 0], params.hgth[center, l, i, 1], params.hgth[center, l, i, 2] ) )
        print('')
def main():

    a = np.eye(3) * 4.0
    atom = 'Si 0 0 0'

    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'sto-3g', 
                 pseudo = 'gth-blyp',
                 verbose = 100,
                 precision = 1.0e-8,
                 dimension = 3)

    cell.build()

    # get plane wave basis information
    basis = plane_wave_basis(cell,
                             ke_cutoff = 500.0 / 27.21138602,
                             n_kpts = [1, 1, 1],
                             use_pseudopotential = True)

    # print GTH parameters
    print_gth_parameters(cell, basis)

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    # print local and non-local components of the pseudopotential matrix
    print('')
    for kid in range (0, len(basis.kpts) ):

        vl = get_gth_pseudopotential(basis, kid, pp_component = 'local')
        vnl = get_gth_pseudopotential(basis, kid, pp_component = 'nonlocal')

        print('GTH pseudopotential (local)')
        print(vl)
        print('')

        print('GTH pseudopotential (nonlocal)')
        print(vnl)
        print('')


if __name__ == "__main__":
    main()
