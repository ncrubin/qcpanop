import numpy as np

from pw_scf import plane_wave_basis
from pw_scf import pw_uks

from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf

def main():

    # define unit cell 
    
    a = np.eye(3) * 4.0
    atom = 'B 0 0 0; H 0 0 2'
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'cc-pvqz', 
                 pseudo = 'gth-blyp',
                 verbose = 100,
                 ke_cutoff = 10000 / 27.21138602,
                 precision = 1.0e-8,
                 #spin = 1,
                 dimension = 3)
    
    cell.build()

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
                             ke_cutoff = 1000.0 / 27.21138602, 
                             n_kpts = [1, 1, 1], 
                             use_pseudopotential = True)
    
    # run pyscf dft
    from pyscf import dft, scf, pbc
    #kmf = pbc.scf.KUHF(cell, kpts = k).run()
    #kmf = pbc.scf.KUKS(cell,xc='lda,', kpts = basis.kpts).run()
    #exit()

   
    # run plane wave scf 
    pw_uks(cell, basis, xc = 'lda')

if __name__ == "__main__":
    main()
