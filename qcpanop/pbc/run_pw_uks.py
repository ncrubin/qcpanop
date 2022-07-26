import numpy as np

from pw_scf import plane_wave_basis
from pw_scf import pw_uks

from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf

import ase
from ase.build import bulk

def main():

    # define unit cell 
    
    #ase_atom = bulk('Si', 'diamond', a = 10.26)
    #ase_atom = bulk('C', 'diamond', a = 6.74)
    #ase_atom = bulk('H', 'diamond', a = 8.88)
    #ase_atom = bulk('Ne', 'diamond', a = 10.26)

    #atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    #a = ase_atom.cell
    
    a = np.eye(3) * 4.0
    atom = 'B 0 0 0; H 0 0 1'
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'gth-dzv', 
                 pseudo = 'gth-blyp',
                 verbose = 100,
                 #ke_cutoff = ke_cutoff,
                 precision = 1.0e-8,
                 #spin = 1,
                 dimension = 3)
    
    cell.build()
    
    # run pyscf dft
    from pyscf import dft, scf, pbc
    #kmf = pbc.scf.KUHF(cell, kpts = k).run()
    #kmf = pbc.scf.KUKS(cell,xc='lda,', kpts = basis.kpts).run()
    #exit()

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
                             ke_cutoff = 500.0 / 27.21138602, 
                             n_kpts = [1, 1, 1], 
                             use_pseudopotential = True)
   
    # run plane wave scf 
    pw_uks(cell, basis, xc = 'lda')

if __name__ == "__main__":
    main()
