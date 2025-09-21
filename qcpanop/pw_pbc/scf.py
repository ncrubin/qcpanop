"""

plane wave scf

"""

# libxc
import pylibxc

# TODO: this dictionary is incomplete and shouldn't be global
functional_name_dict = {
    'hf' : [None, None],
    'lda' : ['lda_x', None],
    'pbe' : ['gga_x_pbe', 'gga_c_pbe']
} 


import time
import warnings
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from qcpanop.pw_pbc.pseudopotential import get_local_pseudopotential_gth
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements

from qcpanop.pw_pbc.basis import get_miller_indices

from pyscf.pbc import tools

def orthonormalize(C):

    """
    orthonormalize orbitals (plane-wave coefficients).
    :param C: orbitals
    :return C_new: orthonormalized orbitals
    """

    # overlap matrix
    S = C.conj().T @ C

    # diagonalize overlap
    evals, evecs = np.linalg.eigh(S)

    # form S^{-1/2}
    S_inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.conj().T

    # apply to orbitals
    return C @ S_inv_sqrt

def get_exact_exchange_energy(basis, occupied_orbitals, N, C):
    """

    evaluate the exact Hartree-Fock exchange energy, according to

        Ex = - 2 pi / Omega sum_{mn in occ} sum_{g} |Cmn(g)|^2 / |g|^2

    where

        Cmn(g) = FT[ phi_m(r) phi_n*(r) ]

    see JCP 108, 4697 (1998) for more details.

    :param basis: plane wave basis information
    :param occupied_orbitals: a list of occupied orbitals
    :param N: the number of electrons
    :param N: the MO transformation matrix

    :return exchange_energy: the exact Hartree-Fock exchange energy

    """

    # precompute indices

    # FFT grid shape
    grid_shape = basis.real_space_grid_dim  # e.g., (nx, ny, nz)
    
    # Precompute: for each compact G index `myg`, find its flat index in FFT grid ordering
    flat_idx = np.empty(len(basis.g), dtype=np.int64)
    for myg in range(len(basis.g)):
        ix, iy, iz = get_miller_indices(myg, basis)
        flat_idx[myg] = np.ravel_multi_index((ix, iy, iz), grid_shape)

    # precompute FFT[1/|r-r'|/g2]
    inv_g2 = np.zeros_like(basis.g2)
    mask = basis.g2 != 0.0
    inv_g2[mask] = 1.0 / basis.g2[mask]

    # accumulate exchange energy and matrix
    exchange_energy = 0.0

    for i in range(0, N):
        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(occupied_orbitals[j].conj() * occupied_orbitals[i])
            Cij = tmp.ravel()[flat_idx]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij = Cij * inv_g2

            exchange_energy += np.sum( Cij.conj() * Kij )

    return -2.0 * np.pi / basis.omega * exchange_energy


def get_exact_exchange_potential(basis, occupied_orbitals, N, C):
    """

    evaluate the exact Hartree-Fock exchange energy, according to

        Ex = - 2 pi / Omega sum_{mn in occ} sum_{g} |Cmn(g)|^2 / |g|^2

    where

        Cmn(g) = FT[ phi_m(r) phi_n*(r) ]

    see JCP 108, 4697 (1998) for more details.

    :param basis: plane wave basis information
    :param occupied_orbitals: a list of occupied orbitals
    :param N: the number of electrons
    :param N: the MO transformation matrix

    :return exchange_energy: the exact Hartree-Fock exchange energy
    :return exchange_potential: the exact Hartree-Fock exchange potential

    """

    # precompute indices

    # FFT grid shape
    grid_shape = basis.real_space_grid_dim  # e.g., (nx, ny, nz)
    
    # Precompute: for each compact G index `myg`, find its flat index in FFT grid ordering
    flat_idx = np.empty(len(basis.g), dtype=np.int64)
    for myg in range(len(basis.g)):
        ix, iy, iz = get_miller_indices(myg, basis)
        flat_idx[myg] = np.ravel_multi_index((ix, iy, iz), grid_shape)

    # precompute FFT[1/|r-r'|/g2]
    inv_g2 = np.zeros_like(basis.g2)
    mask = basis.g2 != 0.0
    inv_g2[mask] = 1.0 / basis.g2[mask]

    # accumulate exchange energy and matrix
    exchange_energy = 0.0

    for i in range(0, N):
        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(occupied_orbitals[j].conj() * occupied_orbitals[i])
            Cij = tmp.ravel()[flat_idx]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij = Cij * inv_g2

            exchange_energy += np.sum( Cij.conj() * Kij )

    # build exchange matrix in planewave basis
    exchange_matrix = np.zeros((basis.n_plane_waves_per_k[0], basis.n_plane_waves_per_k[0]), dtype='complex128')

    Kij_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

    for i in range(0, basis.n_plane_waves_per_k[0]):
        ii = basis.kg_to_g[0][i]

        # Ki(r) = sum_j Kij(r) phi_j(r)
        Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

        # a planewave basis function
        phi_i = np.zeros(len(basis.g), dtype = 'complex128')
        phi_i[ii] = 1.0
        tmp = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        #for myg in range( len(basis.g) ):
        #    tmp[ get_miller_indices(myg, basis) ] = phi_i[myg]
        tmp.ravel()[flat_idx] = phi_i
        phi_i = np.fft.fftn(tmp)

        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(occupied_orbitals[j].conj() * phi_i)
            Cij = tmp.ravel()[flat_idx]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij_G.ravel()[flat_idx] = Cij * inv_g2 #Kij

            # Kij(r) = FFT^-1[Kij(g)]
            Kij_r = np.fft.fftn(Kij_G)

            # action of K on an occupied orbital, i: 
            # Ki(r) = sum_j Kij(r) phi_j(r)
            Ki_r += Kij_r * occupied_orbitals[j]

        # build a row of the exchange matrix: < G' | K | G''> = FFT( Ki_r )
        row_g = np.fft.ifftn(Ki_r)
        for k in range(0, basis.n_plane_waves_per_k[0]):
            kk = basis.kg_to_g[0][k]
            
            exchange_matrix[k, i] = row_g[ get_miller_indices(kk, basis) ] 

    return -2.0 * np.pi / basis.omega * exchange_energy, -4.0 * np.pi / basis.omega * exchange_matrix

def get_xc_potential(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional):
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho_alpha: the alpha spin density (real space)
    :param rho_beta: the beta spin density (real space)
    :param libxc_x_functional: the exchange functional
    :param libxc_c_functional: the correlation functional
    :return xc_alpha: the exchange-correlation energy (alpha)
    :return xc_beta: the exchange-correlation energy (beta)

    """


    v_xc_alpha = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_beta = np.zeros(len(basis.g), dtype = 'complex128')

    if xc != 'hf' :

        # libxc wants a list of density elements [alpha[0], beta[0], alpha[1], beta[1], etc.]
        combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
        combined_rho[::2] = rho_alpha.ravel(order='C')
        combined_rho[1::2] = rho_beta.ravel(order='C')

        # contracted gradient: del rho . del rho as [aa[0], ab[0], bb[0], aa[1], etc.]
        contracted_gradient = np.zeros( [3 * basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2]] )

        # box size
        a = basis.a
        xdim = np.linalg.norm(a[0]) 
        ydim = np.linalg.norm(a[1]) 
        zdim = np.linalg.norm(a[2]) 

        hx = xdim / basis.real_space_grid_dim[0] 
        hy = ydim / basis.real_space_grid_dim[1] 
        hz = zdim / basis.real_space_grid_dim[2] 

        drho_dx_alpha = np.gradient(rho_alpha, axis=0) / hx
        drho_dy_alpha = np.gradient(rho_alpha, axis=1) / hy
        drho_dz_alpha = np.gradient(rho_alpha, axis=2) / hz

        drho_dx_beta = np.gradient(rho_beta, axis=0) / hx
        drho_dy_beta = np.gradient(rho_beta, axis=1) / hy
        drho_dz_beta = np.gradient(rho_beta, axis=2) / hz

        tmp_aa = drho_dx_alpha * drho_dx_alpha \
               + drho_dy_alpha * drho_dy_alpha \
               + drho_dz_alpha * drho_dz_alpha

        tmp_ab = drho_dx_alpha * drho_dx_beta \
               + drho_dy_alpha * drho_dy_beta \
               + drho_dz_alpha * drho_dz_beta

        tmp_bb = drho_dx_beta * drho_dx_beta \
               + drho_dy_beta * drho_dy_beta \
               + drho_dz_beta * drho_dz_beta

        contracted_gradient = np.array(list(zip(tmp_aa.flatten(), tmp_ab.flatten(), tmp_bb.flatten())))

        inp = {
            "rho" : combined_rho,
            "sigma" : contracted_gradient,
            "lapl" : None,
            "tau" : None
        }

        tmp_alpha = np.zeros_like(rho_alpha)
        tmp_beta = np.zeros_like(rho_beta)

        if libxc_x_functional is not None :

            # compute exchange functional
            ret_x = libxc_x_functional.compute( inp )

            vrho_x = ret_x['vrho']
            tmp_alpha += vrho_x[:, 0].reshape(basis.real_space_grid_dim)
            tmp_beta += vrho_x[:, 1].reshape(basis.real_space_grid_dim)

            if 'vsigma' in ret_x :

                vsigma_x = ret_x['vsigma']

                # unpack vsigma_x
                vsigma_x_aa = vsigma_x[:, 0].reshape(basis.real_space_grid_dim)
                vsigma_x_ab = vsigma_x[:, 1].reshape(basis.real_space_grid_dim)
                vsigma_x_bb = vsigma_x[:, 2].reshape(basis.real_space_grid_dim)

                # additional derivatives involving vsigma_x
                dvsigma_x_aa_a = np.gradient(vsigma_x_aa * drho_dx_alpha, axis=0) / hx \
                               + np.gradient(vsigma_x_aa * drho_dy_alpha, axis=1) / hy \
                               + np.gradient(vsigma_x_aa * drho_dz_alpha, axis=2) / hz

                dvsigma_x_ab_a = np.gradient(vsigma_x_ab * drho_dx_alpha, axis=0) / hx \
                               + np.gradient(vsigma_x_ab * drho_dy_alpha, axis=1) / hy \
                               + np.gradient(vsigma_x_ab * drho_dz_alpha, axis=2) / hz

                dvsigma_x_ab_b = np.gradient(vsigma_x_ab * drho_dx_beta, axis=0) / hx \
                               + np.gradient(vsigma_x_ab * drho_dy_beta, axis=1) / hy \
                               + np.gradient(vsigma_x_ab * drho_dz_beta, axis=2) / hz

                dvsigma_x_bb_b = np.gradient(vsigma_x_bb * drho_dx_beta, axis=0) / hx \
                               + np.gradient(vsigma_x_bb * drho_dy_beta, axis=1) / hy \
                               + np.gradient(vsigma_x_bb * drho_dz_beta, axis=2) / hz

                tmp_alpha -= 2.0 * dvsigma_x_aa_a
                tmp_alpha -= dvsigma_x_ab_b

                tmp_beta -= 2.0 * dvsigma_x_bb_b
                tmp_beta -= dvsigma_x_ab_a

        if libxc_c_functional is not None :

            # compute correlaction functional
            ret_c = libxc_c_functional.compute( inp )
            vrho_c = ret_c['vrho']

            tmp_alpha += vrho_c[:, 0].reshape(basis.real_space_grid_dim)
            tmp_beta += vrho_c[:, 1].reshape(basis.real_space_grid_dim)

            if 'vsigma' in ret_c :

                vsigma_c = ret_c['vsigma']

                # unpack vsigma_c
                vsigma_c_aa = vsigma_c[:, 0].reshape(basis.real_space_grid_dim)
                vsigma_c_ab = vsigma_c[:, 1].reshape(basis.real_space_grid_dim)
                vsigma_c_bb = vsigma_c[:, 2].reshape(basis.real_space_grid_dim)

                # additional derivatives involving vsigma_c
                dvsigma_c_aa_a = np.gradient(vsigma_c_aa * drho_dx_alpha, axis=0) / hx \
                               + np.gradient(vsigma_c_aa * drho_dy_alpha, axis=1) / hy \
                               + np.gradient(vsigma_c_aa * drho_dz_alpha, axis=2) / hz

                dvsigma_c_ab_a = np.gradient(vsigma_c_ab * drho_dx_alpha, axis=0) / hx \
                               + np.gradient(vsigma_c_ab * drho_dy_alpha, axis=1) / hy \
                               + np.gradient(vsigma_c_ab * drho_dz_alpha, axis=2) / hz

                dvsigma_c_ab_b = np.gradient(vsigma_c_ab * drho_dx_beta, axis=0) / hx \
                               + np.gradient(vsigma_c_ab * drho_dy_beta, axis=1) / hy \
                               + np.gradient(vsigma_c_ab * drho_dz_beta, axis=2) / hz

                dvsigma_c_bb_b = np.gradient(vsigma_c_bb * drho_dx_beta, axis=0) / hx \
                               + np.gradient(vsigma_c_bb * drho_dy_beta, axis=1) / hy \
                               + np.gradient(vsigma_c_bb * drho_dz_beta, axis=2) / hz

                tmp_alpha -= 2.0 * dvsigma_c_aa_a
                tmp_alpha -= dvsigma_c_ab_b

                tmp_beta -= 2.0 * dvsigma_c_bb_b
                tmp_beta -= dvsigma_c_ab_a


        # fourier transform v_xc(r)
        tmp_alpha = np.fft.ifftn(tmp_alpha)
        tmp_beta = np.fft.ifftn(tmp_beta)

        # unpack v_xc(g) 
        for myg in range( len(basis.g) ):
            v_xc_alpha[myg] = tmp_alpha[ get_miller_indices(myg, basis) ]
            v_xc_beta[myg] = tmp_beta[ get_miller_indices(myg, basis) ]

    else :

        # TODO: fix for general k
        xc_energy = get_exact_exchange_energy(basis, occupied_orbitals, N, C)

    return v_xc_alpha, v_xc_beta

def get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional): 
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho_alpha: the alpha spin density (real space)
    :param rho_beta: the beta spin density (real space)
    :param libxc_x_functional: the exchange functional
    :param libxc_c_functional: the correlation functional

    """

    xc_energy = 0.0

    # libxc wants a list of density elements [alpha[0], beta[0], alpha[1], beta[1], etc.]
    combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
    combined_rho[::2] = rho_alpha.ravel(order='C')
    combined_rho[1::2] = rho_beta.ravel(order='C')

    # contracted gradient: del rho . del rho as [aa[0], ab[0], bb[0], aa[1], etc.]
    contracted_gradient = None

    # TODO: logic should be updated once we support more functionals
    if libxc_x_functional != 'lda_x' and libxc_c_functional != None :

        # box size
        a = basis.a
        xdim = np.linalg.norm(a[0]) 
        ydim = np.linalg.norm(a[1]) 
        zdim = np.linalg.norm(a[2]) 
        
        hx = xdim / basis.real_space_grid_dim[0]
        hy = ydim / basis.real_space_grid_dim[1]
        hz = zdim / basis.real_space_grid_dim[2]

        drho_dx_alpha = np.gradient(rho_alpha, axis=0) / hx
        drho_dy_alpha = np.gradient(rho_alpha, axis=1) / hy
        drho_dz_alpha = np.gradient(rho_alpha, axis=2) / hz

        drho_dx_beta = np.gradient(rho_beta, axis=0) / hx
        drho_dy_beta = np.gradient(rho_beta, axis=1) / hy
        drho_dz_beta = np.gradient(rho_beta, axis=2) / hz

        tmp_aa = drho_dx_alpha * drho_dx_alpha \
               + drho_dy_alpha * drho_dy_alpha \
               + drho_dz_alpha * drho_dz_alpha

        tmp_ab = drho_dx_alpha * drho_dx_beta \
               + drho_dy_alpha * drho_dy_beta \
               + drho_dz_alpha * drho_dz_beta

        tmp_bb = drho_dx_beta * drho_dx_beta \
               + drho_dy_beta * drho_dy_beta \
               + drho_dz_beta * drho_dz_beta

        contracted_gradient = np.array(list(zip(tmp_aa.flatten(), tmp_ab.flatten(), tmp_bb.flatten())))

    inp = {
        "rho" : combined_rho,
        "sigma" : contracted_gradient,
        "lapl" : None,
        "tau" : None
    }

    val = np.zeros_like(rho_alpha.flatten())

    # compute exchange functional
    if libxc_x_functional is not None :
        ret_x = libxc_x_functional.compute( inp, do_vxc = False )
        zk_x = ret_x['zk']
        val += (rho_alpha.flatten() + rho_beta.flatten() ) * zk_x.flatten()

    # compute correlation functional
    if libxc_c_functional is not None :
        ret_c = libxc_c_functional.compute( inp, do_vxc = False )
        zk_c = ret_c['zk']
        val += (rho_alpha.flatten() + rho_beta.flatten() ) * zk_c.flatten()

    xc_energy = val.sum() * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) )

    return xc_energy

def get_matrix_elements(basis, kid, vg):

    """

    unpack the potential matrix for given k-point: <G'|V|G''> = V(G'-G'')

    :param basis: plane wave basis information
    :param kid: index for a given k-point
    :param vg: full potential container in the plane wave basis for the density (up to E <= 2 x ke_cutoff)
    :return potential: the pseudopotential matrix in the plane wave basis for the orbitals, for this k-point (up to E <= ke_cutoff)

    """

    potential = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype='complex128')

    gkind = basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]

    for aa in range(basis.n_plane_waves_per_k[kid]):

        ik = basis.kg_to_g[kid][aa]
        gdiff = basis.miller[ik] - basis.miller[gkind[aa:]] + np.array(basis.reciprocal_max_dim)
        #inds = basis.miller_to_g[gdiff.T.tolist()]
        # inds = basis.miller_to_g[tuple(gdiff.T.tolist())]
        inds = basis.miller_to_g[gdiff[:, 0], 
                                 gdiff[:, 1],
                                 gdiff[:, 2]]

        potential[aa, aa:] = vg[inds]

    return potential

def get_nuclear_electronic_potential(cell, basis, valence_charges = None):

    """
    v(r) = \sum_{G}v(G)exp(iG.r)

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param valence_charges: the charges associated with the valence
    :return: vneG: the nuclear-electron potential in the plane wave basis
    """

    # nuclear charge of atoms in cell. 
    charges = cell.atom_charges()
    if valence_charges is not None:
        charges = valence_charges

    assert basis.SI.shape[1] == basis.g.shape[0]

    # this is Z_{I} * S_{I}
    rhoG = np.dot(charges, basis.SI)  

    coulG = np.divide(4.0 * np.pi, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

    vneG = - rhoG * coulG / basis.omega

    return vneG

def form_orbital_gradient(basis, C, N, F, kid):
    """

    form density matrix and orbital gradient from fock matrix and orbital coefficients

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param N: the number of electrons
    :param F: the fock matrix
    :param kid: index for a given k-point

    :return orbital_gradient: the orbital gradient

    """

    tmporbs = np.zeros([basis.n_plane_waves_per_k[kid], N], dtype = 'complex128')
    for pp in range(N):
        tmporbs[:, pp] = C[:, pp]

    #D = np.einsum('ik,jk->ij', tmporbs.conj(), tmporbs)
    D = np.matmul(tmporbs, tmporbs.conj().T) 

    # only upper triangle of F is populated ... symmetrize and rescale diagonal
    #F = F + F.conj().T
    #diag = np.diag(F)
    #np.fill_diagonal(F, 0.5 * diag)
    #for pp in range(basis.n_plane_waves_per_k[kid]):
    #    F[pp][pp] *= 0.5

    #orbital_gradient = np.einsum('ik,kj->ij', F, D)
    #orbital_gradient -= np.einsum('ik,kj->ij', D, F)
    orbital_gradient = np.matmul(F, D)
    orbital_gradient -= np.matmul(D, F)

    return orbital_gradient

def get_density(basis, C, Ne, Nmo, kid):
    """

    get real-space density from molecular orbital coefficients

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param Ne: the number of electrons
    :param Nmo: the number of molecular orbitals (Nmo >= Ne)
    :param kid: index for a given k-point

    :return rho: the density
    :return orbitals_r: the molecular orbitals in real space 

    """

    orbitals_r = []
    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    for pp in range(Nmo):

        occ = np.zeros(basis.real_space_grid_dim,dtype = 'complex128')

        for tt in range( basis.n_plane_waves_per_k[kid] ):

            ik = basis.kg_to_g[kid][tt]
            occ[ get_miller_indices(ik, basis) ] = C[tt, pp]

        occ = np.fft.fftn(occ) 
        orbitals_r.append(occ)

        if pp < Ne:
            rho += np.absolute(occ)**2.0 / basis.omega

    return ( 1.0 / len(basis.kpts) ) * rho, orbitals_r

def form_fock_matrix(basis, kid, v = None): 
    """

    form fock matrix

    :param basis: plane wave basis information
    :param kid: index for a given k-point
    :param v: the potential ( coulomb + xc + electron-nucleus / local pp )

    :return fock: the fock matrix

    """

    fock = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128')

    # unpack potential
    if v is not None:
        fock += get_matrix_elements(basis, kid, v)
    
    # get non-local part of the pseudopotential (not jellium)
    if basis.use_pseudopotential and not jellium:
        fock += get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)

    # get kinetic energy
    kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]
    T = np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0 
    diagonals = T + fock.diagonal()
    np.fill_diagonal(fock, diagonals)

    # only upper triangle of fock is populated ... symmetrize and rescale diagonal
    fock = fock + fock.conj().T
    diag = np.diag(fock)
    np.fill_diagonal(fock, 0.5 * diag)

    return fock

def get_one_electron_energy(basis, C, N, kid, v_ne = None, jellium = False):
    """

    get one-electron part of the energy

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param N: the number of electrons
    :param kid: index for a given k-point
    :param v_ne: nuclear electron potential or local part of the pseudopotential

    :return one_electron_energy: the one-electron energy

    """

    # oei = T + V 
    oei = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128')

    # jellium
    if not jellium:
        if v_ne is not None:
            oei += get_matrix_elements(basis, kid, v_ne)

        if basis.use_pseudopotential:
            oei += get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)

    kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]
    T = np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0 
    diagonals = T + oei.diagonal()
    np.fill_diagonal(oei, diagonals)

    oei = oei + oei.conj().T

    diag = np.diag(oei)
    np.fill_diagonal(oei, 0.5 * diag)

    tmporbs = np.zeros([basis.n_plane_waves_per_k[kid], N], dtype = 'complex128')
    for pp in range(N):
        tmporbs[:, pp] = C[:, pp]

    one_electron_energy = np.einsum('pi,pq,qi->',tmporbs.conj(), oei, tmporbs) / len(basis.kpts)

    return one_electron_energy

def get_coulomb_energy(basis, C, N, kid, v_coulomb):
    """

    get the coulomb contribution to the energy

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param N: the number of electrons
    :param kid: index for a given k-point
    :param v_coulomb: the coulomb potential

    :return coulomb_energy: the coulomb contribution to the energy

    """

    # oei = 1/2 J
    oei = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128')
    oei = get_matrix_elements(basis, kid, v_coulomb)

    # only upper triangle of oei is populated ... symmetrize and rescale diagonal
    oei = oei + oei.conj().T
    
    diag = np.diag(oei)
    np.fill_diagonal(oei, 0.5 * diag)

    tmporbs = np.zeros([basis.n_plane_waves_per_k[kid], N], dtype = 'complex128')
    for pp in range(N):
        tmporbs[:, pp] = C[:, pp]

    coulomb_energy = 0.5 * np.einsum('pi,pq,qi->',tmporbs.conj(), oei, tmporbs) / len(basis.kpts)

    return coulomb_energy

def uks(cell, basis, 
        xc = 'lda', 
        guess_mix = False, 
        e_convergence = 1e-8, 
        d_convergence = 1e-6, 
        diis_dimension = 8, 
        damp_density = True, 
        damping_iterations = 8,
        ace_exchange = True,
        jellium = False,
        jellium_ne = 2,
        maxiter=500):

    """

    plane wave unrestricted kohn-sham

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param xc: the exchange-correlation functional
    :param guess_mix: do mix alpha homo and lumo to break spin symmetry?
    :param e_convergence: the convergence in the energy
    :param d_convergence: the convergence in the orbital gradient
    :param damp_density: do dampen density?
    :param damping_iterations: for how many iterations should we dampen the fock matrix
    :param maxiter: maximum number of scf iterations
    :return total energy
    :return Calpha: alpha MO coefficients
    :return Cbeta: beta MO coefficients

    """
 
    print('')
    print('    ************************************************')
    print('    *                                              *')
    print('    *                Plane-wave UKS                *')
    print('    *                                              *')
    print('    ************************************************')
    print('')

    if xc != 'lda' and xc != 'pbe' and xc != 'hf' :
        raise Exception("uks only supports xc = 'lda' and 'hf' for now")

    libxc_x_functional = None
    libxc_c_functional = None

    if functional_name_dict[xc][0] is not None :
        libxc_x_functional = pylibxc.LibXCFunctional(functional_name_dict[xc][0], "polarized")

    if functional_name_dict[xc][1] is not None :
        libxc_c_functional = pylibxc.LibXCFunctional(functional_name_dict[xc][1], "polarized")

    # get nuclear repulsion energy
    enuc = cell.energy_nuc()

    # coulomb and xc potentials in reciprocal space
    v_coulomb = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_alpha = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_beta = np.zeros(len(basis.g), dtype = 'complex128')

    # density in reciprocal space
    rhog = np.zeros(len(basis.g), dtype = 'complex128')
    inner_rhog = np.zeros(len(basis.g), dtype = 'complex128')

    # charges
    valence_charges = cell.atom_charges()
    
    if basis.use_pseudopotential: 
        for i in range (0,len(valence_charges)):
            valence_charges[i] = int(basis.gth_params[i].Zion)

    # electron-nucleus potential
    v_ne = None
    if not basis.use_pseudopotential: 
        v_ne = get_nuclear_electronic_potential(cell, basis, valence_charges = valence_charges)
    else :
        v_ne = get_local_pseudopotential_gth(basis)

    # madelung correction
    madelung = tools.pbc.madelung(cell, basis.kpts)

    nalpha, nbeta = cell.nelec
    total_charge = cell.charge

    # jellium
    if jellium:
        v_ne *= 0.0
        nalpha = jellium_ne // 2
        nbeta = jellium_ne // 2
        enuc = 0.0

    # damp fock matrix (helps with convergence sometimes)
    damping_factor = 1.0
    if damp_density :
        damping_factor = 0.8

    # diis 
    diis_start_cycle = damping_iterations

    rs = (3 * basis.omega / ( 4.0 * np.pi * (nalpha + nbeta)))**(1.0/3.0)
    print("")
    print('    Wigner-Seitz radius (rs)                     %20.12f' % (rs))
    print('    exchange functional:                         %20s' % ( functional_name_dict[xc][0] ) )
    print('    correlation functional:                      %20s' % ( functional_name_dict[xc][1] ) )
    print('    e_convergence:                               %20.2e' % ( e_convergence ) )
    print('    d_convergence:                               %20.2e' % ( d_convergence ) )
    print('    no. k-points:                                %20i' % ( len(basis.kpts) ) )
    print('    KE cutoff (eV)                               %20.2f' % ( basis.ke_cutoff * 27.21138602 ) )
    print('    no. basis functions (orbitals, gamma point): %20i' % ( basis.n_plane_waves_per_k[0] ) )
    print('    no. basis functions (density):               %20i' % ( len(basis.g) ) )
    print('    total_charge:                                %20i' % ( total_charge ) )
    print('    no. alpha bands:                             %20i' % ( nalpha ) )
    print('    no. beta bands:                              %20i' % ( nbeta ) )
    print('    break spin symmetry:                         %20s' % ( "yes" if guess_mix is True else "no" ) )
    print('    damp density:                                %20s' % ( "yes" if damp_density is True else "no" ) )
    print('    no. damping iterations:                      %20i' % ( damping_iterations ) )
    #print('    diis start iteration:                        %20i' % ( diis_start_cycle ) )
    print('    no. diis vectors:                            %20i' % ( diis_dimension ) )

    if guess_mix :
        print("")
        print("    ==> WARNING <==")
        print("")
        print("        guess_mix = True is not working and is currently disabled")
 

    print("")
    print("    ==> Begin UKS Iterations <==")
    print("")

    if jellium:
        print("    %5s %20s %20s %20s %10s %20s %20s %20s %20s %20s" % ('iter', 'energy', '|dE|', '||[F, D]||', 'Nelec', '||Ca - Cb||', 'kinetic', 'coulomb', 'exchange', 'madelung'))
    else :
        print("    %5s %20s %20s %20s %10s" % ('iter', 'energy', '|dE|', '||[F, D]||', 'Nelec'))

    old_total_energy = 0.0

    Calpha = []
    Cbeta = []

    epsilon_alpha = []
    epsilon_beta = []

    # for ace
    Binv_alpha_ace = []
    Binv_beta_ace = []

    Ki_alpha = []
    Ki_beta = []

    # nmo ... number of desired molecular orbitals ... must be at least ne
    # warning: ace has trouble for nmo > ne with eigsh, but lobpcg seems to work
    nmo_alpha = nalpha #+ 1
    nmo_beta = nbeta #+ 1
    #if nbeta > nalpha:  
    #    nmo = nbeta

    for kid in range ( len(basis.kpts) ):

        Calpha.append(np.random.rand(basis.n_plane_waves_per_k[kid], nmo_alpha) * 1e-3)
        Cbeta.append(np.random.rand(basis.n_plane_waves_per_k[kid], nmo_beta) * 1e-3)
        #Calpha.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo), dtype='complex128'))
        #Cbeta.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo), dtype='complex128'))
        for i in range (nmo_alpha):
            Calpha[kid][i, i] = 1.0
        for i in range (nmo_beta):
            Cbeta[kid][i, i] = 1.0
        Calpha[kid] = orthonormalize(Calpha[kid])
        #Cbeta[kid] = orthonormalize(Cbeta[kid])
        Cbeta[kid] = Calpha[kid].copy()


        Binv_alpha_ace.append(np.zeros((nmo_alpha, nmo_alpha), dtype='complex128'))
        Binv_beta_ace.append(np.zeros((nmo_beta, nmo_beta), dtype='complex128'))

        Ki_alpha.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_alpha), dtype='complex128'))
        Ki_beta.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_beta), dtype='complex128'))

        epsilon_alpha.append(np.zeros((nmo_alpha), dtype='complex128'))
        epsilon_beta.append(np.zeros((nmo_beta), dtype='complex128'))

    # diis extrapolates fock matrix
    from pyscf import lib
    diis = lib.diis.DIIS()
    diis.space = diis_dimension

    # begin UKS iterations
    xc_energy = 0.0
    scf_iter = 0 # initialize variable incase maxiter = 0
    one_electron_energy = 0.0
    coulomb_energy = 0.0
    recompute_exchange = True

    # precompute indices

    # FFT grid shape
    grid_shape = basis.real_space_grid_dim  # e.g., (nx, ny, nz)
    
    # for each compact G index `myg`, find its flat index in FFT grid ordering
    flat_idx = np.empty(len(basis.g), dtype=np.int64)
    for myg in range(len(basis.g)):
        ix, iy, iz = get_miller_indices(myg, basis)
        flat_idx[myg] = np.ravel_multi_index((ix, iy, iz), grid_shape)

    # precompute linear grid indices for each k-point
    grid_idx_k = []
    for kid in range ( len(basis.kpts) ):
        ijk = np.array([get_miller_indices(ik, basis) for ik in basis.kg_to_g[kid]])
        coords = tuple(ijk.T)
        grid_idx_k.append(np.ravel_multi_index(coords, grid_shape))

    # precompute FFT[1/|r-r'|/g2]
    inv_g2 = np.zeros_like(basis.g2)
    mask = basis.g2 != 0.0
    inv_g2[mask] = 1.0 / basis.g2[mask]

    # so we have occ_alpha and occ_beta arrays ... won't work for kpts > 0
    my_rho_a, occ_alpha = get_density(basis, Calpha[0], nalpha, nmo_alpha, kid)
    my_rho_b, occ_beta = get_density(basis, Cbeta[0], nbeta, nmo_beta, kid)

    # kinetic energy
    T = []
    for kid in range ( len(basis.kpts) ):
        kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]
        T.append(np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0)

        #epsilon_alpha, Calpha[kid] = scipy.linalg.eigh(np.diag(T[kid]), eigvals=(0, nmo-1))
        #Calpha[kid] = orthonormalize(Calpha[kid])
        #Cbeta[kid] = Calpha[kid].copy()

    #print(2*np.sum(np.sort(T[0])[:nalpha]))
    #exit()

    v_alpha = 0 * v_ne
    v_beta = 0 * v_ne
    v_alpha_old = 0 * v_ne
    v_beta_old = 0 * v_ne

    rho_alpha = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    rho_alpha_old = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    rho_beta_old = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

    # jellium guess
    rho_alpha = nalpha / basis.omega * np.ones(basis.real_space_grid_dim, dtype = 'float64')
    rho_beta = nbeta / basis.omega * np.ones(basis.real_space_grid_dim, dtype = 'float64')

    # rebuild potentials
    rho = rho_alpha + rho_beta

    # coulomb potential
    tmp = np.fft.ifftn(rho)
    for myg in range( len(basis.g) ):
        inner_rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

    v_coulomb = 4.0 * np.pi * np.divide(inner_rhog, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

    # jellium ... but only true in the thermodynamic limit
    if jellium:
        v_coulomb *= 0.0

    # exchange-correlation potential
    if xc != 'hf' :

        v_xc_alpha, v_xc_beta = get_xc_potential(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)
        xc_energy = get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)

    v_alpha = v_coulomb + v_xc_alpha
    v_beta = v_coulomb + v_xc_beta

    for scf_iter in range(maxiter):

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        v_alpha = v_coulomb + v_xc_alpha
        v_beta = v_coulomb + v_xc_beta

        v_alpha_old = v_alpha
        v_beta_old = v_beta

        # potential in real space
        v_alpha_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        v_alpha_r.ravel()[flat_idx] = v_alpha + v_ne
        v_alpha_r = np.fft.fftn(v_alpha_r).real

        v_beta_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        v_beta_r.ravel()[flat_idx] = v_beta + v_ne
        v_beta_r = np.fft.fftn(v_beta_r).real

        # zero density for this iteration
        rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        # damping factor
        damp = 1.0
        if scf_iter < diis_start_cycle and damp_density : 
            damp = damping_factor

        # orbital gradient
        error_vector = np.zeros(0)

        Kij_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

        for kid in range( len(basis.kpts) ):

            Vnl = get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)
            Vnl = Vnl + Vnl.conj().T
            diag = np.diag(Vnl)
            np.fill_diagonal(Vnl, 0.5 * diag)

            # jellium
            if jellium:
                Vnl *= 0.0

            Fa_c = np.zeros((basis.n_plane_waves_per_k[kid], nmo_alpha), dtype='complex128')
            exchange_alpha = np.zeros((basis.n_plane_waves_per_k[kid], nmo_alpha), dtype='complex128')
            for i in range (nmo_alpha):

                # exchange
                Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
                if xc == 'hf':
                    for j in range(0, nalpha):

                        # Cij(r') = phi_i(r') phi_j*(r')
                        # Cij(g) = FFT[Cij(r')]
                        tmp = np.fft.ifftn(occ_alpha[j].conj() * occ_alpha[i])
                        Cij = tmp.ravel()[flat_idx]

                        # Kij(g) = Cij(g) * FFT[1/|r-r'|]
                        Kij_G.ravel()[flat_idx] = Cij * inv_g2 #Kij

                        # Kij(r) = FFT^-1[Kij(g)]
                        Kij_r = np.fft.fftn(Kij_G)

                        # action of K on an occupied orbital, i: 
                        # Ki(r) = sum_j Kij(r) phi_j(r)
                        Ki_r += Kij_r * occ_alpha[j]

                    Ki_r *= -4.0 * np.pi / basis.omega

                # action of potential on orbitals in real space, then transform to reciprocal space
                tmp = v_alpha_r * occ_alpha[i] #+ Ki_r # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis

                Fa_c[:,i] = T[kid] * Calpha[kid][:, i] + tmp[basis.kg_to_g[kid]] # map last term to small flattened basis

                # for ace
                tmp = Ki_r # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                Ki_alpha[kid][:,i] = tmp[basis.kg_to_g[kid]] # reciprocal space, small flattened basis

                # non-ace exchange
                tmp = Ki_r # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                exchange_alpha[:,i] = tmp[basis.kg_to_g[kid]] # map last term to small flattened basis

            Fb_c = np.zeros((basis.n_plane_waves_per_k[kid], nmo_beta), dtype='complex128')
            exchange_beta = np.zeros((basis.n_plane_waves_per_k[kid], nmo_beta), dtype='complex128')
            for i in range (nmo_beta):

                # exchange
                Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
                if xc == 'hf':
                    for j in range(0, nbeta):

                        # Cij(r') = phi_i(r') phi_j*(r')
                        # Cij(g) = FFT[Cij(r')]
                        tmp = np.fft.ifftn(occ_beta[j].conj() * occ_beta[i])
                        Cij = tmp.ravel()[flat_idx]

                        # Kij(g) = Cij(g) * FFT[1/|r-r'|]
                        Kij_G.ravel()[flat_idx] = Cij * inv_g2 #Kij

                        # Kij(r) = FFT^-1[Kij(g)]
                        Kij_r = np.fft.fftn(Kij_G)

                        # action of K on an occupied orbital, i: 
                        # Ki(r) = sum_j Kij(r) phi_j(r)
                        Ki_r += Kij_r * occ_beta[j]

                    Ki_r *= -4.0 * np.pi / basis.omega 

                # action of potential on orbitals in real space, then transform to reciprocal space
                tmp = v_beta_r * occ_beta[i] #+ Ki_r  # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                Fb_c[:,i] = T[kid] * Cbeta[kid][:, i] + tmp[basis.kg_to_g[kid]] # map last term to small flattened basis

                # for ace
                tmp = Ki_r  # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                Ki_beta[kid][:,i] = tmp[basis.kg_to_g[kid]] # reciprocal space, small flattened basis

                # non-ace exchange
                tmp = Ki_r # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                exchange_beta[:,i] = tmp[basis.kg_to_g[kid]] # map last term to small flattened basis

            if xc == 'hf':

                # for ace < phi_i | Kj>^{-1}
                tmp = -Calpha[kid][:, :nmo_alpha].conj().T @ Ki_alpha[kid]
                L = np.linalg.cholesky(tmp)

                # how's our cholesky decomposition looking?
                assert (np.allclose(tmp, L @ L.conj().T))

                Linv = np.linalg.inv(L)
                Binv_alpha_ace[kid] = -Linv.conj().T @ Linv

                # how's our inverse looking?
                assert (np.allclose(-tmp @ Binv_alpha_ace[kid], np.eye(tmp.shape[0])))

                tmp = -Cbeta[kid][:, :nmo_beta].conj().T @ Ki_beta[kid]
                L = np.linalg.cholesky(tmp)

                # how's our cholesky decomposition looking?
                assert (np.allclose(tmp, L @ L.conj().T))

                Linv = np.linalg.inv(L)
                Binv_beta_ace[kid] = -Linv.conj().T @ Linv

                # how's our inverse looking?
                assert (np.allclose(-tmp @ Binv_beta_ace[kid], np.eye(tmp.shape[0])))

                # for ace

                # (< phi_j | K) | c >
                tmp = Ki_alpha[kid].conj().T @ Calpha[kid][:, :nmo_alpha]
                # sum_j Binv_{ij} < phi_j | K | c > 
                tmp = Binv_alpha_ace[kid] @ tmp 
                # sum_i K | phi_i >  Binv_{ij} < phi_j | K | c >
                ace_alpha = Ki_alpha[kid] @ tmp 

                # < phi_j | K | c >
                tmp = Ki_beta[kid].conj().T @ Cbeta[kid][:, :nmo_beta]
                # sum_j Binv_{ij} < phi_j | K | c > 
                tmp = Binv_beta_ace[kid] @ tmp 
                # sum_i K | phi_i >  Binv_{ij} < phi_j | K | c >
                ace_beta = Ki_beta[kid] @ tmp 

                # is ace representation equivalent to original exact exchange representation?
                assert (np.allclose(ace_alpha, exchange_alpha))
                assert (np.allclose(ace_beta, exchange_beta))

                Fa_c += ace_alpha
                Fb_c += ace_beta
            
            Fa_c += Vnl @ Calpha[kid][:, :nmo_alpha]
            Fb_c += Vnl @ Cbeta[kid][:, :nmo_beta]

            #grad_a = Fa_c - epsilon_alpha[kid][np.newaxis, :nalpha] * Calpha[kid][:,:nalpha]
            #grad_b = Fb_c - epsilon_beta[kid][np.newaxis, :nbeta] * Cbeta[kid][:,:nbeta]

            c_Fa_c = Calpha[kid][:, :nmo_alpha].conj().T @ Fa_c
            c_Fb_c = Cbeta[kid][:, :nmo_beta].conj().T @ Fb_c
            grad_a = Fa_c - Calpha[kid][:,:nmo_alpha] @ c_Fa_c
            grad_b = Fb_c - Cbeta[kid][:,:nmo_beta] @ c_Fb_c

            error_vector = np.hstack( (error_vector, grad_a.flatten(), grad_b.flatten() ) )

        # norm of orbital gradient
        conv = np.linalg.norm(error_vector)

        # damp or extrapolate density or potential
        rho_alpha_old = rho_alpha.copy()
        rho_beta_old = rho_beta.copy()
        if scf_iter < diis_start_cycle:

            # damping?
            #v_alpha = (1.0-damp) * v_alpha + damp * v_alpha_old
            #v_beta = (1.0-damp) * v_beta + damp * v_beta_old
            rho_alpha = (1.0-damp) * rho_alpha + damp * rho_alpha_old
            rho_beta = (1.0-damp) * rho_beta + damp * rho_beta_old


        solution_vector = np.hstack( (rho_alpha.flatten(), rho_beta.flatten()) )
        #solution_vector = np.hstack( (v_alpha, v_beta) )
        new_solution_vector = diis.update(solution_vector, error_vector)
        #v_alpha = new_solution_vector[:len(v_alpha)]
        #v_beta = new_solution_vector[len(v_alpha):]
        rho_alpha = new_solution_vector[:len(solution_vector)//2].reshape(rho_alpha_old.shape)
        rho_beta = new_solution_vector[len(solution_vector)//2:].reshape(rho_beta_old.shape)

        rho_alpha.clip(min = 0)
        rho_beta.clip(min = 0)

        # rebuild potentials
        rho = rho_alpha + rho_beta

        # coulomb potential
        tmp = np.fft.ifftn(rho)
        for myg in range( len(basis.g) ):
            inner_rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

        v_coulomb = 4.0 * np.pi * np.divide(inner_rhog, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

        # jellium ... but only true in the thermodynamic limit
        if jellium:
            v_coulomb *= 0.0

        # exchange-correlation potential
        if xc != 'hf' :

            v_xc_alpha, v_xc_beta = get_xc_potential(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)
            xc_energy = get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)

        v_alpha = v_coulomb + v_xc_alpha
        v_beta = v_coulomb + v_xc_beta
    
        # potential in real space
        v_alpha_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        v_alpha_r.ravel()[flat_idx] = v_alpha + v_ne
        v_alpha_r = np.fft.fftn(v_alpha_r).real

        v_beta_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        v_beta_r.ravel()[flat_idx] = v_beta + v_ne
        v_beta_r = np.fft.fftn(v_beta_r).real

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        rho_alpha = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
        rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        # diagonalize fock matrix with extrapolated potential
        Kij_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

        for kid in range ( len(basis.kpts) ):

            Vnl = get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)
            Vnl = Vnl + Vnl.conj().T
            diag = np.diag(Vnl)
            np.fill_diagonal(Vnl, 0.5 * diag)

            # jellium
            if jellium:
                Vnl *= 0.0

            def apply_fock_operator_to_orbital(c):
                """
                apply fock operator to an orbital. note that the function depends on 
                some parameters not passed in as argumnets

                :param c: orbital in reciprococal space
                :return F @ c: action of fock operator on the orbital in reciprococal space

                :implicit parameter my_N: the number of electrons of the same spin as orbital c 
                :implicit parameter occ_list: a list of occupied orbitals in real space with the same spin as orbital c
                :implicit parameter my_v_r: the potential in real space experienced by orbital c (excluding exchange)
                """

                # orbital in real space
                #occ = np.zeros(np.prod(grid_shape), dtype=np.complex128) # eigsh
                occ = np.zeros([np.prod(grid_shape), c.shape[1]], dtype=np.complex128) # lobpcg
                occ[grid_idx_k[kid]] = c
                occ = occ.reshape(grid_shape)
                occ = np.fft.fftn(occ) 

                # exchange
                Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
                if xc == 'hf' and not ace_exchange:
                    for j in range(0, my_N):

                        # Cij(r') = phi_i(r') phi_j*(r')
                        # Cij(g) = FFT[Cij(r')]
                        tmp = np.fft.ifftn(occ_list[j].conj() * occ)
                        Cij = tmp.ravel()[flat_idx]

                        # Kij(g) = Cij(g) * FFT[1/|r-r'|]
                        Kij_G.ravel()[flat_idx] = Cij * inv_g2 #Kij

                        # Kij(r) = FFT^-1[Kij(g)]
                        Kij_r = np.fft.fftn(Kij_G)

                        # action of K on an occupied orbital, i: 
                        # Ki(r) = sum_j Kij(r) phi_j(r)
                        Ki_r += Kij_r * occ_list[j]

                    Ki_r *= -4.0 * np.pi / basis.omega

                # action of potential on orbitals in real space, then transform to reciprocal space
                tmp = my_v_r * occ  # real space, 3d
                tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                tmp = tmp[basis.kg_to_g[kid]] # reciprocal space, small flattened basis
                tmp = tmp.reshape(c.shape) # for lobpcg

                #F_c = T[kid] * c + tmp + Vnl @ c # eigsh
                F_c = T[kid][:, None] * c + tmp + Vnl @ c # lobpcg

                # ace

                # (< phi_j | K) | c >
                tmp = my_Ki.conj().T @ c
                # sum_j Binv_{ij} < phi_j | K | c > 
                tmp = my_Binv @ tmp 
                # sum_i K | phi_i >  Binv_{ij} < phi_j | K | c >
                ace = my_Ki @ tmp 

                if ace_exchange :
                    F_c += ace

                # exact exchange
                if not ace_exchange :
                    tmp = Ki_r # real space, 3d
                    tmp = np.fft.ifftn(tmp) # reciprocal space, 3d
                    tmp = tmp.ravel()[flat_idx] # reciprocal space, large flattened basis
                    tmp = tmp[basis.kg_to_g[kid]] # reciprocal space, small flattened basis
                    tmp = tmp.reshape(c.shape) # for lobpcg

                    F_c += tmp

                #print(np.linalg.norm(ace - tmp))

                return F_c

            F_C = LinearOperator((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), matvec=apply_fock_operator_to_orbital, dtype='complex128')

            if not jellium:
                my_N = nalpha
                occ_list = occ_alpha.copy()
                my_v_r = v_alpha_r.copy()
                my_Ki = Ki_alpha[kid].copy()
                my_Binv = Binv_alpha_ace[kid].copy()
                epsilon_alpha[kid], Calpha[kid] = scipy.sparse.linalg.lobpcg(F_C, Calpha[kid], largest=False, maxiter=2000, tol=d_convergence*0.1)

                my_N = nbeta
                occ_list = occ_beta.copy()
                my_v_r = v_beta_r.copy()
                my_Ki = Ki_beta[kid].copy()
                my_Binv = Binv_beta_ace[kid].copy()
                epsilon_beta[kid], Cbeta[kid] = scipy.sparse.linalg.lobpcg(F_C, Cbeta[kid], largest=False, maxiter=2000, tol=d_convergence*0.1)

            else :
                epsilon_alpha[kid], Calpha[kid] = scipy.linalg.eigh(np.diag(T[kid]), eigvals=(0, nalpha))
                #epsilon_beta[kid], Cbeta[kid] = scipy.linalg.eigh(fock_b[kid], eigvals=(0, nbeta))

            # break spin symmetry? # TODO this is broken, which probably indicates there is some other problem ...
            #if guess_mix is True and scf_iter == 0:

            #    c = np.cos(0.05 * np.pi)
            #    s = np.sin(0.05 * np.pi)

            #    tmp1 = c * Calpha[kid][:, nalpha-1] - s * Calpha[kid][:, nalpha]
            #    tmp2 = s * Calpha[kid][:, nalpha-1] + c * Calpha[kid][:, nalpha]

            #    Calpha[kid][:, nalpha-1] = tmp1
            #    Calpha[kid][:, nalpha] = tmp2

            # why do i need to orthonormalize my orbitals???
            Calpha[kid] = orthonormalize(Calpha[kid])
            Cbeta[kid] = orthonormalize(Cbeta[kid])
            if jellium:
                Cbeta[kid] = Calpha[kid].copy()

            # update density
            my_rho_alpha, occ_alpha = get_density(basis, Calpha[kid], nalpha, nmo_alpha, kid)
            my_rho_beta, occ_beta = get_density(basis, Cbeta[kid], nbeta, nmo_beta, kid)

            # density should be non-negative ...
            rho_alpha += my_rho_alpha.clip(min = 0)
            rho_beta += my_rho_beta.clip(min = 0)

            # one-electron part of the energy (alpha)
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Calpha[kid], 
                                                           nalpha, 
                                                           kid, 
                                                           v_ne = v_ne,
                                                           jellium = jellium)

            # one-electron part of the energy (beta)
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Cbeta[kid], 
                                                           nbeta, 
                                                           kid, 
                                                           v_ne = v_ne,
                                                           jellium = jellium)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Calpha[kid], nalpha, kid, v_coulomb)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Cbeta[kid], nbeta, kid, v_coulomb)

        rho = rho_alpha + rho_beta

        # coulomb potential
        tmp = np.fft.ifftn(rho)
        for myg in range( len(basis.g) ):
            inner_rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

        v_coulomb = 4.0 * np.pi * np.divide(inner_rhog, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

        # jellium ... but only true in the thermodynamic limit
        if jellium:
            v_coulomb *= 0.0

        # exchange-correlation potential
        if xc != 'hf' :

            v_xc_alpha, v_xc_beta = get_xc_potential(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)
            xc_energy = get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)

        else :

            # exact exchange energy
            xc_energy = get_exact_exchange_energy(basis, occ_alpha, nalpha, Calpha)
            if nbeta > 0:
                my_xc_energy = get_exact_exchange_energy(basis, occ_beta, nbeta, Cbeta)
                xc_energy += my_xc_energy

            # jellium
            if not jellium:
                xc_energy -= 0.5 * (nalpha + nbeta) * madelung

        # total energy
        new_total_energy = np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc
        if jellium:
            new_total_energy -= 0.5 * (nalpha + nbeta) * madelung

        # convergence in energy
        energy_diff = np.abs(new_total_energy - old_total_energy)

        # update energy
        old_total_energy = new_total_energy

        # charge
        charge = ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.absolute(rho))

        if jellium:
            print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf %20.12f %20.12f %20.12f %20.12f %20.12f" %  ( scf_iter, new_total_energy, energy_diff, conv, charge, np.linalg.norm(Calpha[kid] - Cbeta[kid]), np.real(one_electron_energy), np.real(coulomb_energy), np.real(xc_energy), -0.5 * (nalpha + nbeta) * madelung))
        else :
            print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf" %  ( scf_iter, new_total_energy, energy_diff, conv, charge))

        if conv < d_convergence and energy_diff < e_convergence :
            break

    if scf_iter == maxiter - 1:
        print('')
        print('    UKS iterations did not converge.')
        print('')
    else:
        print('')
        print('    UKS iterations converged!')
        print('')


    print('    ==> energy components <==')
    print('')
    print('    nuclear repulsion energy: %20.12lf' % ( enuc ) )
    print('    one-electron energy:      %20.12lf' % ( np.real(one_electron_energy) ) )
    print('    coulomb energy:           %20.12lf' % ( np.real(coulomb_energy) ) )
    print('    xc energy:                %20.12lf' % ( np.real(xc_energy) ) )
    if jellium:
        print('    Madelung:                 %20.12lf' % ( -0.5 * (nalpha + nbeta) * madelung) )
    print('')
    if jellium:
        print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc -0.5 * (nalpha + nbeta) * madelung ) )
    else :
        print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc ) )
    print('')

    return new_total_energy, Calpha, Cbeta

