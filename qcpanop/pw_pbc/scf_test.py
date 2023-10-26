"""
test functions in pw-code
"""
import numpy as np

import ase
from ase.build import bulk

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc import gto, scf


from qcpanop.pw_pbc.basis import plane_wave_basis


def test_format_density_for_libxc():
    """Test the reshape functionality that maps
       this code's representation of the density to 
       that expected by libxc
    """
    ase_atom = bulk('C', 'diamond', a = 6.74)
    #ase_atom = bulk('H', 'diamond', a = 8.88)
    #ase_atom = bulk('Ne', 'diamond', a = 10.26)

    atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    a = ase_atom.cell 
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'cc-pvqz',
                 pseudo = 'gth-blyp',
                 verbose = 0,
                 ke_cutoff = 5000 / 27.21138602,
                 precision = 1.0e-8,
                 charge = 0,
                 spin = 0,
                 dimension = 3)
    
    cell.build()

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
                             ke_cutoff = 1000.0 / 27.21138602, 
                             n_kpts = [1, 1, 1],
                             nl_pp_use_legendre = True)


    rho_alpha = np.random.randn(*basis.real_space_grid_dim[:3])
    rho_beta = np.random.randn(*basis.real_space_grid_dim[:3])

    # libxc wants a list of density elements [alpha[0], beta[0], alpha[1], beta[1], etc.]
    combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
    assert np.isclose(2 * basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2],
                      2 * np.prod(basis.real_space_grid_dim[:3]))

    count = 0
    for i in range (0, basis.real_space_grid_dim[0] ):
        for j in range (0, basis.real_space_grid_dim[1] ):
            for k in range (0, basis.real_space_grid_dim[2] ):
                combined_rho[count] = rho_alpha[i, j, k]
                combined_rho[count + 1] = rho_beta[i, j, k]
                count = count + 2
    
    flattened_rho_alpha = rho_alpha.ravel(order='C')    
    flattened_rho_beta = rho_beta.ravel(order='C')
    test_combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
    test_combined_rho[::2] = flattened_rho_alpha
    test_combined_rho[1::2] = flattened_rho_beta
    assert np.allclose(test_combined_rho, combined_rho)

    # Now test going back from vrho
    # column 0 is alpha, column 1 is beta
    vrho = np.random.randn(np.prod(basis.real_space_grid_dim[:3]), 2)
    test_tmp_alpha = vrho[:, 0].reshape(basis.real_space_grid_dim)
    test_tmp_beta = vrho[:, 1].reshape(basis.real_space_grid_dim)
    true_tmp_alpha = np.zeros_like(rho_alpha)
    true_tmp_beta = np.zeros_like(rho_beta)
    count = 0
    for i in range (0, basis.real_space_grid_dim[0] ):
        for j in range (0, basis.real_space_grid_dim[1] ):
            for k in range (0, basis.real_space_grid_dim[2] ):
                true_tmp_alpha[i, j, k] = vrho[count, 0]
                true_tmp_beta[i, j, k] = vrho[count, 1]
                count = count + 1
    assert np.allclose(test_tmp_alpha, true_tmp_alpha)
    assert np.allclose(test_tmp_beta, true_tmp_beta)





if __name__ == "__main__":
    test_format_density_for_libxc()