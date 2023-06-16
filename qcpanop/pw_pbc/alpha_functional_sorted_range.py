import itertools
import numpy as np
from qcpanop.pw_pbc.basis import plane_wave_basis, get_miller_indices
from qcpanop.pw_pbc.scf import uks
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements, get_spherical_harmonics_and_projectors_gth


from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf
from pyscf.pbc import gto as pbcgto


import ase
from ase.build import bulk
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import matplotlib.pyplot as plt


def main():

    # define unit cell 
    
    #ase_atom = bulk('Si', 'diamond', a = 10.26)
    # ase_atom = bulk('C', 'diamond', a = 6.74)
    # ase_atom = bulk('H', 'diamond', a = 8.88)
    ase_atom = bulk('Ne', 'diamond', a = 10.26)

    atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    a = ase_atom.cell 
    ke_cutoff = 3000
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'cc-pvqz',
                 pseudo = 'gth-lda',
                 verbose = 100,
                 ke_cutoff = ke_cutoff / 27.21138602, 
                 precision = 1.0e-8,
                 charge = 0,
                 spin = 0,
                 dimension = 3)
    
    cell.build()

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
                             ke_cutoff = ke_cutoff / 27.21138602,  # 1000 eV cutoff
                             n_kpts = [1, 1, 1],
                             nl_pp_use_legendre = True)
    b = cell.reciprocal_vectors()
    
    ################################################################################
    #
    # Now evaluate the cost function 
    #  alpha[d, l, i, j] = max_{q}|F^{i}_{l}(||k_{q+d}||) F^{j}_{l}(||k_{q}||)|
    # for gamma point for the basis defined as all k < ke_cutoff
    # This is smaller than the full_basis!
    #
    #################################################################################

    kid = 0 # id of k-point. zero for gamma
    gkind = basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]] # list of indices for basis functions for this k-point
    gk = basis.g[gkind] # basis functions for this k-point
    gk_miller = basis.miller[gkind] # miller_indices of basis

    ###################
    #
    # DEBUGGING CHECK
    #
    #####################
    # Check if everything makes sense
    for ii in range(len(gk_miller)):
        m1, m2, m3 = gk_miller[ii]
        test_g = m1 * b[0] + m2 * b[1] + m3 * b[2]
        assert np.allclose(test_g, gk[ii])

    gmax = len(gk)
    lmax = basis.gth_params[0].lmax
    i_j_max = basis.gth_params[0].imax
    print(vars(basis.gth_params[0]))

    # d, l, i, j
    alpha = np.zeros((gmax, lmax, i_j_max, i_j_max), dtype=np.complex128)
    f_li = np.zeros((lmax, i_j_max, gmax))
    norm_kq = np.sqrt(np.einsum('ik,ik->i', gk, gk))
    for l in range(lmax):
        for i in range(i_j_max):
            f_li[l, i, :] = pbcgto.pseudo.pp.projG_li(norm_kq, l, i, 1) # basis.gth_params[0].rl[l])

    # alpha[d, l, i, j] = max_{q}|F^{i}_{l}(||k_{q+d}||) F^{j}_{l}(||k_{q}||)|
    f_dli = np.zeros((gmax, lmax, i_j_max, gmax))

    sorted_gk = np.argsort(np.einsum('ni,ni->n', gk, gk))

    for d in range(len(gk_miller)):
        # d_miller_index = gk_miller[d]
        # k_q_d_miller_indices = d_miller_index + gk_miller
        # assert np.allclose(gk[d], d_miller_index[0] * b[0] + d_miller_index[1] * b[1] + d_miller_index[2] * b[2]  )
        kgd_summed = gk + gk[sorted_gk[d]]
        norm_kqd = np.sqrt(np.einsum('ik,ik->i', kgd_summed, kgd_summed))
        for l in range(lmax):
            for j in range(i_j_max):
                f_dli[d, l, j, :] = pbcgto.pseudo.pp.projG_li(norm_kqd, l, i, 1)

    # Now compute alpha

    for d in range(len(gk_miller)):
        for l in range(lmax):
            for i, j in itertools.product(range(i_j_max), repeat=2):
                alpha[d, l, i, j] = np.max(np.abs(np.multiply(f_dli[d, l, i, :], f_li[l, j, :])))

    fig, ax = plt.subplots(nrows=1, ncols=1)    
    for l in range(lmax):
        for i, j in itertools.product(range(i_j_max), repeat=2):
            ax.plot(range(len(gk_miller)), alpha[:, l, i, j], label='l={},i={},j={}'.format(l, i, j))

    ax.set_xlabel("d", fontsize=14)
    ax.set_ylabel(r"$\alpha(d, l, i, j)$", fontsize=14)
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim([0, 10])
    # plt.savefig("alpha_vs_d_C_gth_LDA_pyscf_sorted_range.png", format='PNG', dpi=300)

    plt.show() 

if __name__ == "__main__":
    main()