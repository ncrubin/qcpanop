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
from qcpanop.pw_pbc.pseudopotential import nonlocal_pseudopotential_on_orbitals

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

def get_exact_exchange_energy(basis, phi_r, N, C, occupation_numbers):
    """

    evaluate the exact Hartree-Fock exchange energy, according to

        Ex = - 2 pi / Omega sum_{mn in occ} sum_{g} |Cmn(g)|^2 / |g|^2

    where

        Cmn(g) = FT[ phi_m(r) phi_n*(r) ]

    see JCP 108, 4697 (1998) for more details.

    :param basis: plane wave basis information
    :param phi_r: a list of occupied orbitals in real space
    :param N: the number of orbitals (could be greater than number of electrons with the use of the occupation_numbers list)
    :param C: the MO transformation matrix
    :param occupation_numbers: a list of occupation numbers (0, 1, or fermi-dirac for smearing)

    :return exchange_energy: the exact Hartree-Fock exchange energy

    """

    # accumulate exchange energy and matrix
    exchange_energy = 0.0

    for i in range(0, N):
        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(phi_r[j].conj() * phi_r[i]) * occupation_numbers[i] * occupation_numbers[j]
            Cij = tmp.ravel()[basis.flat_idx]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij = Cij * basis.inv_g2

            exchange_energy += np.sum( Cij.conj() * Kij )

    return -2.0 * np.pi / basis.omega * exchange_energy

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

    coulG = 4.0 * np.pi * basis.inv_g2

    vneG = - rhoG * coulG / basis.omega

    return vneG

def get_density(basis, C, Ne, Nmo, kid, occupation_numbers):
    """

    get real-space density from molecular orbital coefficients

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param Ne: the number of electrons
    :param Nmo: the number of molecular orbitals (Nmo >= Ne)
    :param kid: index for a given k-point
    :param occupation_numbers: a list of occupation numbers (0, 1, or fermi-dirac for smearing)

    :return rho: the density
    :return phi_r: the molecular orbitals in real space 

    """

    phi_r = []
    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    for pp in range(Nmo):

        phi = np.zeros(basis.real_space_grid_dim,dtype = 'complex128')

        for tt in range( basis.n_plane_waves_per_k[kid] ):

            ik = basis.kg_to_g[kid][tt]
            phi[ get_miller_indices(ik, basis) ] = C[tt, pp]

        phi = np.fft.fftn(phi) 
        phi_r.append(phi)

        rho += np.absolute(phi)**2.0 / basis.omega * occupation_numbers[pp]

    return ( 1.0 / len(basis.kpts) ) * rho, phi_r

def fock_on_orbitals(basis, kid, ne, nmo, phi_r, C, T, v_r, xc, occupation_numbers, jellium):
    """
    evaluate action of fock matrix on orbitals and build ace operator

    :param basis: the plane wave basis object
    :param kid: the current k-point
    :param ne: number of electrons
    :param nmo: number of bands (could be larger than ne)
    :param phi_r: orbitals, in real space
    :param C: orbitals, in reciprocal space
    :param T: diagonal of the kinetic energy matrix, in reciprocal space
    :param v_r: the potential (coulomb + local pseudopotential + xc), in real space
    :param xc: the exchange-correlation functional
    :param occupation_numbers: a list of occupation numbers (0, 1, or fermi-dirac for smearing)
    :param jellium: is this jellium? should we worry about the pseudopotential?

    :return F_c: the action of the fock matrix on the orbials
    :return Ki: exchange operator acting on orbitals, i, in reciprocal space
    :return exchange: the exchange matrix
    """

    Kij_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    Ki = np.zeros((basis.n_plane_waves_per_k[kid], nmo), dtype='complex128')
    F_c = np.zeros((basis.n_plane_waves_per_k[kid], nmo), dtype='complex128')
    exchange = np.zeros((basis.n_plane_waves_per_k[kid], nmo), dtype='complex128')
    for i in range (nmo):

        # exchange
        Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        if xc == 'hf':
            for j in range(0, nmo): # note, this has changed from ne to nmo with the introduction of occupation_numbers

                # Cij(r') = phi_i(r') phi_j*(r')
                # Cij(g) = FFT[Cij(r')]
                tmp = np.fft.ifftn(phi_r[j].conj() * phi_r[i])
                Cij = tmp.ravel()[basis.flat_idx]

                # Kij(g) = Cij(g) * FFT[1/|r-r'|]
                Kij_G[:] = 0.0
                Kij_G.ravel()[basis.flat_idx] = Cij * basis.inv_g2 #Kij

                # Kij(r) = FFT^-1[Kij(g)]
                Kij_r = np.fft.fftn(Kij_G)

                # action of K on an occupied orbital, i: 
                # Ki(r) = sum_j Kij(r) phi_j(r)
                Ki_r += Kij_r * phi_r[j] * occupation_numbers[j]

            Ki_r *= -4.0 * np.pi / basis.omega

        # action of potential on orbitals in real space, then transform to reciprocal space
        tmp_vphi = v_r * phi_r[i] #+ Ki_r # real space, 3d
        tmp_vphi = np.fft.ifftn(tmp_vphi) # reciprocal space, 3d
        tmp_vphi = tmp_vphi.ravel()[basis.flat_idx] # reciprocal space, large flattened basis

        F_c[:,i] = T[kid] * C[kid][:, i] + tmp_vphi[basis.kg_to_g[kid]] # map last term to small flattened basis

        # for ace
        tmp_Ki = Ki_r.copy() # real space, 3d
        tmp_Ki = np.fft.ifftn(tmp_Ki) # reciprocal space, 3d
        tmp_Ki = tmp_Ki.ravel()[basis.flat_idx] # reciprocal space, large flattened basis
        Ki[:,i] = tmp_Ki[basis.kg_to_g[kid]] # reciprocal space, small flattened basis

        # non-ace exchange
        tmp_exch = Ki_r.copy() # real space, 3d
        tmp_exch = np.fft.ifftn(tmp_exch) # reciprocal space, 3d
        tmp_exch = tmp_exch.ravel()[basis.flat_idx] # reciprocal space, large flattened basis
        exchange[:,i] = tmp_exch[basis.kg_to_g[kid]] # map last term to small flattened basis

    if not jellium and basis.use_pseudopotential: 
        F_c += nonlocal_pseudopotential_on_orbitals(basis, kid, C[kid])

    return F_c, Ki, exchange

def fock_on_orbitals_using_ace(basis, kid, ne, nmo, phi_r, c, T, v_r, xc, Ki, B_ace, ace_exchange, occupation_numbers, jellium):
    """
    apply fock operator to a set of orbitals using the ace operator

    :param basis: the plane wave basis object
    :param kid: the current k-point
    :param ne: number of electrons
    :param nmo: number of bands (could be larger than ne)
    :param phi_r: orbitals, in real space
    :param C: orbitals, in reciprocal space
    :param T: diagonal of the kinetic energy matrix, in reciprocal space
    :param v_r: the potential (coulomb + local pseudopotential + xc), in real space
    :param xc: the exchange-correlation functional
    :param Ki: exchange operator acting on orbitals, i, in reciprocal space
    :param B: ACE B matrix
    :param ace_exchange: do use the ace operator for exchange?
    :param occupation_numbers: a list of occupation numbers (0, 1, or fermi-dirac for smearing)
    :param jellium: is this jellium? should we worry about the pseudopotential?

    :return F @ c: action of fock operator on the orbitals in reciprococal space
    """

    # orbital in real space
    current_phi = np.zeros([np.prod(basis.real_space_grid_dim), c.shape[1]], dtype=np.complex128) # lobpcg
    current_phi[basis.grid_idx_k[kid]] = c
    current_phi = current_phi.reshape(basis.real_space_grid_dim)
    current_phi = np.fft.fftn(current_phi) 

    # exchange
    Kij_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    if xc == 'hf' and not ace_exchange:
        for j in range(0, nmo): # note, this has changed from ne to nmo with the introduction of occupation_numbers

            # Cij(r') = phi_i(r') phi_j*(r')
            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(phi_r[j].conj() * current_phi)
            Cij = tmp.ravel()[basis.flat_idx]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij_G[:] = 0.0
            Kij_G.ravel()[basis.flat_idx] = Cij * basis.inv_g2 #Kij

            # Kij(r) = FFT^-1[Kij(g)]
            Kij_r = np.fft.fftn(Kij_G)

            # action of K on an occupied orbital, i: 
            # Ki(r) = sum_j Kij(r) phi_j(r)
            Ki_r += Kij_r * phi_r[j] * occupation_numbers[j]

        Ki_r *= -4.0 * np.pi / basis.omega

    # action of potential on orbitals in real space, then transform to reciprocal space
    tmp_vphi = v_r * current_phi  # real space, 3d
    tmp_vphi = np.fft.ifftn(tmp_vphi) # reciprocal space, 3d
    tmp_vphi = tmp_vphi.ravel()[basis.flat_idx] # reciprocal space, large flattened basis
    tmp_vphi = tmp_vphi[basis.kg_to_g[kid]] # reciprocal space, small flattened basis
    tmp_vphi = tmp_vphi.reshape(c.shape) # for lobpcg

    #F_c = T[kid] * c + tmp # eigsh
    F_c = T[kid][:, None] * c + tmp_vphi  # lobpcg

    if not jellium and basis.use_pseudopotential: 
        F_c += nonlocal_pseudopotential_on_orbitals(basis, kid, c)

    if xc == 'hf':
        if ace_exchange :

            # (< phi_j | K) | c >
            tmp = Ki.conj().T @ c
            # sum_j B_{ij} < phi_j | K | c > 
            tmp = B_ace @ tmp 
            # sum_i K | phi_i >  B_{ij} < phi_j | K | c >
            ace = Ki @ tmp 

            F_c += ace
 
        else :

            tmp_exch = Ki_r # real space, 3d
            tmp_exch = np.fft.ifftn(tmp_exch) # reciprocal space, 3d
            tmp_exch = tmp_exch.ravel()[basis.flat_idx] # reciprocal space, large flattened basis
            tmp_exch = tmp_exch[basis.kg_to_g[kid]] # reciprocal space, small flattened basis
            tmp_exch = tmp_exch.reshape(c.shape) # for lobpcg
            F_c += tmp_exch

    #print(np.linalg.norm(ace - tmp))

    return F_c

def build_B_ace(ne, nmo, C, Ki, exchange):
    """
    build B matrix for ACE
    see text after Eq. 13 of J. Chem. Theory Comput. 12, 2242-2249 (2016).

    :param ne: number of electrons
    :param nmo: number of bands (could be greater than ne)
    :param C: orbital coefficients in reciprocal space
    :param Ki: action of exchange operator on orbitals in reciprocal space
    :param exchange: exchange contribution to fock matrix (exact, for testing ace)

    :return B_ace: the B matrix for ACE
    """

    # for ace < phi_i | Kj>^{-1}
    tmp = -C[:, :nmo].conj().T @ Ki
    L = np.linalg.cholesky(tmp)

    # how's our cholesky decomposition looking?
    assert (np.allclose(tmp, L @ L.conj().T))

    Linv = np.linalg.inv(L)
    B_ace = -Linv.conj().T @ Linv

    # how's our inverse looking?
    assert (np.allclose(-tmp @ B_ace, np.eye(tmp.shape[0])))

    # test ACE representation of exchange

    # (< phi_j | K) | c >
    tmp = Ki.conj().T @ C[:, :nmo]
    # sum_j B_{ij} < phi_j | K | c > 
    tmp = B_ace @ tmp 
    # sum_i K | phi_i >  B_{ij} < phi_j | K | c >
    ace = Ki @ tmp

    assert (np.allclose(ace, exchange))

    return B_ace

def compute_local_potentials(rho_alpha, rho_beta, v_ne, basis, xc, libxc_x_functional, libxc_c_functional, jellium):
    """
    compute local potentials (coulomb + local pseudopotential + xc)

    :param rho_alpha: the alpha-spin density
    :param rho_beta: the beta-spin density
    :param v_ne: the external potential or local pseudopotential
    :param basis: the plane wave basis object
    :param xc: the name of the exchange-correlation potential
    :param libxc_x_functional: the libxc exchange functional
    :param libxc_c_functional: the libxc correlation functional
    :param jellium: are we modeling jellium?

    :return v_coulomb: the coulomb potential in reciprocal space
    :return v_alpha_r: the alpha-spin local potential in real space
    :return v_beta_r: the beta-spin local potential in real space
    """

    # coulomb potential
    tmp = np.fft.ifftn(rho_alpha + rho_beta)
    rhog = np.zeros(len(basis.g), dtype = 'complex128')
    for myg in range( len(basis.g) ):
        rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

    v_coulomb = 4.0 * np.pi * rhog * basis.inv_g2

    # jellium ... but only true in the thermodynamic limit ... or if we want the uniform density
    #if jellium:
    #    v_coulomb *= 0.0

    # alpha- and beta-spin local potentials
    v_alpha = v_coulomb.copy()
    v_beta = v_coulomb.copy()

    # exchange-correlation potential
    if xc != 'hf' :

        v_xc_alpha, v_xc_beta = get_xc_potential(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)
        xc_energy = get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)

        v_alpha += v_xc_alpha
        v_beta +=  v_xc_beta

    # alpha-spin potential in real space
    v_alpha_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    v_alpha_G.ravel()[basis.flat_idx] = v_alpha + v_ne
    v_alpha_r = np.fft.fftn(v_alpha_G).real

    # beta-spin potential in real space
    v_beta_G = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    v_beta_G.ravel()[basis.flat_idx] = v_beta + v_ne
    v_beta_r = np.fft.fftn(v_beta_G).real

    return v_coulomb, v_alpha_r, v_beta_r

def fermi_dirac(ne, eps, mu, kBT = None):
    """
    occupations from fermi–dirac distribution for given chemical potential

    :param ne: number of electrons
    :param eps: orbital energies
    :param mu: chemical potential
    :param kBT: boltzman factor times temperature

    :return fermi-diract distribution
    """
    if kBT == None:
        ret = np.zeros_like(eps).real
        ret[:ne] = 1.0
        return ret

    x = (eps - mu) / kBT
    return 1.0 / (np.exp(x) + 1.0)

def find_chemical_potential(ne, eps, kBT = None, tol=1e-10, maxit = 200):
    """
    bisection search for chemical potential, mu, for fermi-dirac distribution:

    sum_i f_i = ne

    :param ne: number of electrons
    :param eps: orbital energies
    :param kBT: boltzman factor times temperature
    :param tol: convergence threshold for bisection search
    :param maxit: maximum number of iterations

    """

    if kBT == None:
        return kBT

    # bracket mu between min and max eigenvalue
    mu_lo = np.min(eps) - 10.0 * kBT
    mu_hi = np.max(eps) + 10.0 * kBT

    # bisection search
    for it in range(maxit):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        n_mid = np.sum(fermi_dirac(ne, eps, mu_mid, kBT = kBT))
        if n_mid > ne:
            mu_hi = mu_mid
        else:
            mu_lo = mu_mid
        if abs(mu_hi - mu_lo) < tol:
            break
    if it == maxit:
        raise Exception('bisection search for the chemical potential failed.')

    return mu_mid

def uks(cell, basis, 
        xc = 'lda', 
        guess_mix = False, 
        e_convergence = 1e-8, 
        d_convergence = 1e-6, 
        diis_dimension = 8, 
        ace_exchange = True,
        jellium = False,
        jellium_ne = 2,
        maxiter=500,
        kBT=None,
        print_level = 1):

    """

    plane wave unrestricted kohn-sham

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param xc: the exchange-correlation functional
    :param guess_mix: do mix alpha homo and lumo to break spin symmetry?
    :param e_convergence: the convergence in the energy
    :param d_convergence: the convergence in the orbital gradient
    :param maxiter: maximum number of scf iterations
    :return total energy
    :return Calpha: alpha MO coefficients
    :return Cbeta: beta MO coefficients
    :return kBT: boltzman factor times temperature (for smearing)

    """

    if print_level > 0:
        print('')
        print('    ************************************************')
        print('    *                                              *')
        print('    *                Plane-wave UKS                *')
        print('    *                                              *')
        print('    ************************************************')
        print('')

    if len(basis.kpts) > 1 and kBT is not None:
        raise Exception("smearing only works for the gamma point for now")

    if len(basis.kpts) > 1 and xc == 'hf':
        raise Exception("exact exchange only works for the gamma point for now")

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

    # density in reciprocal space
    rhog = np.zeros(len(basis.g), dtype = 'complex128')

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

    # nmo ... number of desired molecular orbitals ... must be at least ne
    nmo_alpha = nalpha
    nmo_beta = nbeta
    #if guess_mix:
    #    nmo_alpha += 1
    #    nmo_beta += 1
    
    # increase nmo if we're doing smearing
    if kBT is not None:
        nmo_alpha *= 2
        nmo_beta *= 2

    if print_level > 0:
        print("")
    if jellium:
        rs = (3 * basis.omega / ( 4.0 * np.pi * (nalpha + nbeta)))**(1.0/3.0)
        if print_level > 0:
            print('    Wigner-Seitz radius (rs)                     %20.12f' % (rs))
    if kBT is not None:
        if print_level > 0:
            print('    kBT for smearing (eV)                        %20.12f' % (kBT * 27.21138602))
    if print_level > 0:
        print('    exchange functional:                         %20s' % ( functional_name_dict[xc][0] ) )
        print('    correlation functional:                      %20s' % ( functional_name_dict[xc][1] ) )
        print('    use pseudopotential:                         %20s' % ( "yes" if basis.use_pseudopotential is True else "no" ) )
        print('    e_convergence:                               %20.2e' % ( e_convergence ) )
        print('    d_convergence:                               %20.2e' % ( d_convergence ) )
        print('    no. k-points:                                %20i' % ( len(basis.kpts) ) )
        print('    KE cutoff (eV)                               %20.2f' % ( basis.ke_cutoff * 27.21138602 ) )
        print('    no. basis functions (orbitals, gamma point): %20i' % ( basis.n_plane_waves_per_k[0] ) )
        print('    no. basis functions (density):               %20i' % ( len(basis.g) ) )
        print('    omega (a0^3):                                %20.2f' % ( basis.omega ) )
        print('    total_charge:                                %20i' % ( total_charge ) )
        print('    no. alpha occupied bands:                    %20i' % ( nalpha ) )
        print('    no. beta occupied bands:                     %20i' % ( nbeta ) )
        print('    no. total alpha bands:                       %20i' % ( nmo_alpha ) )
        print('    no. total beta bands:                        %20i' % ( nmo_beta ) )
        if xc == 'hf' or jellium:
            print('    madelung contribution to Ex:                 %20.12f' % ( -0.5 * (nalpha + nbeta) * madelung ) )
        print('    break spin symmetry:                         %20s' % ( "yes" if guess_mix is True else "no" ) )
        print('    no. diis vectors:                            %20i' % ( diis_dimension ) )

    if guess_mix :
        if print_level > 0:
            print("")
            print("    ==> WARNING <==")
            print("")
            print("        guess_mix = True is not working and is currently disabled")

    old_total_energy = 0.0

    Calpha = []
    Cbeta = []

    epsilon_alpha = []
    epsilon_beta = []

    # for ace
    B_alpha_ace = []
    B_beta_ace = []

    # for ace
    Ki_alpha = []
    Ki_beta = []

    # occupation numbers
    occ_num_alpha = np.ones(nmo_alpha, dtype=np.float64)
    occ_num_beta = np.ones(nmo_beta, dtype=np.float64)
    occ_num_alpha[nalpha:] = 0.0
    occ_num_beta[nbeta:] = 0.0

    # kinetic energy
    T = []
    for kid in range ( len(basis.kpts) ):
        kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]
        T.append(np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0)

    for kid in range ( len(basis.kpts) ):

        #Calpha.append(np.random.rand(basis.n_plane_waves_per_k[kid], nmo_alpha) * 1e-4)
        #Cbeta.append(np.random.rand(basis.n_plane_waves_per_k[kid], nmo_beta) * 1e-4)
        Calpha.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_alpha), dtype='complex128'))
        Cbeta.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_beta), dtype='complex128'))
        for i in range (nmo_alpha):
            Calpha[kid][i, i] = 1.0
        for i in range (nmo_beta):
            Cbeta[kid][i, i] = 1.0
        Calpha[kid] = orthonormalize(Calpha[kid])
        Cbeta[kid] = orthonormalize(Cbeta[kid])

        B_alpha_ace.append(np.zeros((nmo_alpha, nmo_alpha), dtype='complex128'))
        B_beta_ace.append(np.zeros((nmo_beta, nmo_beta), dtype='complex128'))

        Ki_alpha.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_alpha), dtype='complex128'))
        Ki_beta.append(np.zeros((basis.n_plane_waves_per_k[kid], nmo_beta), dtype='complex128'))

        epsilon_alpha.append(np.zeros((nmo_alpha), dtype='complex128'))
        epsilon_beta.append(np.zeros((nmo_beta), dtype='complex128'))

        # jellium orbital guess, plus some noise
        if jellium:

            lowest_energy_T_index = np.argsort(T[kid])[:nalpha]
            epsilon_alpha[kid] = T[kid][lowest_energy_T_index]
            Calpha[kid] = np.zeros((len(T[kid]), nalpha), dtype=np.complex128)
            for orb_idx, pw_idx in enumerate(lowest_energy_T_index):
                Calpha[kid][pw_idx, orb_idx] = 1.0

            #eps_a, ca = np.linalg.eigh(np.diag(T[kid]))
            #epsilon_alpha[kid], Calpha[kid] = eps_a[:nmo_alpha], ca[:, :nmo_alpha]

            #Calpha[kid] += np.random.rand(basis.n_plane_waves_per_k[kid], nmo_alpha) * 1e-3
            Calpha[kid] = orthonormalize(Calpha[kid])
            Cbeta[kid] = Calpha[kid].copy()

            # update density
            my_rho_alpha, phi_alpha = get_density(basis, Calpha[kid], nalpha, nmo_alpha, kid, occ_num_alpha)
            my_rho_beta, phi_beta = get_density(basis, Cbeta[kid], nbeta, nmo_beta, kid, occ_num_beta)

            # density should be non-negative ...
            rho_alpha = my_rho_alpha.clip(min = 0).real
            rho_beta = my_rho_beta.clip(min = 0).real

            # kinetic energy part of the one-electron energy
            one_electron_energy = np.sum(np.einsum('pi,p->i', np.abs(Calpha[kid][:,:nmo_alpha])**2 * occ_num_alpha, T[kid]))
            one_electron_energy += np.sum(np.einsum('pi,p->i', np.abs(Cbeta[kid][:,:nmo_beta])**2 * occ_num_beta, T[kid]))

            # exact exchange energy
            xc_energy = get_exact_exchange_energy(basis, phi_alpha, nmo_alpha, Calpha, occ_num_alpha)
            if nbeta > 0:
                my_xc_energy = get_exact_exchange_energy(basis, phi_beta, nmo_beta, Cbeta, occ_num_beta)
                xc_energy += my_xc_energy
            energy = one_electron_energy + xc_energy - 0.5 * (nalpha + nbeta) * madelung

            if print_level > 0:
                print()
                print("    %20s %20s %20s %20s" % ('energy', 'kinetic', 'exchange', 'madelung'))
                print("    %20.12lf %20.12lf %20.12lf %20.12f" %  (energy.real, one_electron_energy.real, xc_energy.real, -0.5 * (nalpha + nbeta) * madelung))
                print()
                exit()
            print("%5i\t%6.2f\t%20.12e\t%20.12e\t%20.12e\t%20.12e\t%20.12e" %  (nalpha + nbeta, rs, basis.omega, energy.real, one_electron_energy.real, xc_energy.real, -0.5 * (nalpha + nbeta) * madelung))
            return energy.real, Calpha, Cbeta

    # initialize phi_alpha and phi_beta arrays ... won't work for kpts > 0
    rho_alpha, phi_alpha = get_density(basis, Calpha[0], nalpha, nmo_alpha, kid, occ_num_alpha)
    rho_beta, phi_beta = get_density(basis, Cbeta[0], nbeta, nmo_beta, kid, occ_num_beta)

    rho_alpha_old = rho_alpha.copy()
    rho_beta_old = rho_beta.copy()

    # local potentials
    v_coulomb, v_alpha_r, v_beta_r = compute_local_potentials(rho_alpha, rho_beta, v_ne, basis, xc, libxc_x_functional, libxc_c_functional, jellium)

    # diis extrapolates the alpha- and beta-spin densities
    from pyscf import lib
    diis = lib.diis.DIIS()
    diis.space = diis_dimension

    if print_level > 0:
        print("")
        print("    ==> Begin UKS Iterations <==")
        print("")

    if jellium:
        if print_level > 0:
            print("    %5s %20s %20s %20s %10s %20s %20s %20s %20s %20s" % ('iter', 'energy', '|dE|', '||[F, D]||', 'Nelec', '||Ca - Cb||', 'kinetic', 'coulomb', 'exchange', 'madelung'))
    else :
        if print_level > 0:
            print("    %5s %20s %20s %20s %10s" % ('iter', 'energy', '|dE|', '||[F, D]||', 'Nelec'))

    # begin UKS iterations
    xc_energy = 0.0
    scf_iter = 0 # initialize variable incase maxiter = 0
    one_electron_energy = 0.0
    coulomb_energy = 0.0

    for scf_iter in range(maxiter):

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        # damping
        damp = 0.5
        if scf_iter > 0:
            rho_alpha = rho_alpha_old * (1.0 - damp) + rho_alpha * damp
            rho_beta = rho_beta_old * (1.0 - damp) + rho_beta * damp

            # compute local potentials
            v_coulomb, v_alpha_r, v_beta_r = compute_local_potentials(rho_alpha, rho_beta, v_ne, basis, xc, libxc_x_functional, libxc_c_functional, jellium)

        # evaluate the orbital gradient
        error_vector = np.zeros(0)
        for kid in range( len(basis.kpts) ):

            Fa_c, Ki_alpha[kid], exchange_alpha = fock_on_orbitals(basis, kid, nalpha, nmo_alpha, phi_alpha, Calpha, T, v_alpha_r, xc, occ_num_alpha, jellium)
            Fb_c, Ki_beta[kid], exchange_beta = fock_on_orbitals(basis, kid, nbeta, nmo_beta, phi_beta, Cbeta, T, v_beta_r, xc, occ_num_beta, jellium)

            if xc == 'hf':

                B_alpha_ace[kid] = build_B_ace(nalpha, nmo_alpha, Calpha[kid], Ki_alpha[kid], exchange_alpha)
                B_beta_ace[kid] = build_B_ace(nbeta, nmo_beta, Cbeta[kid], Ki_beta[kid], exchange_beta)

                Fa_c += exchange_alpha
                Fb_c += exchange_beta
            
            #grad_a = Fa_c - epsilon_alpha[kid][np.newaxis, :nmo_alpha] * Calpha[kid][:,:nmo_alpha]
            #grad_b = Fb_c - epsilon_beta[kid][np.newaxis, :nmo_beta] * Cbeta[kid][:,:nmo_beta]

            c_Fa_c = Calpha[kid][:, :nmo_alpha].conj().T @ Fa_c
            c_Fb_c = Cbeta[kid][:, :nmo_beta].conj().T @ Fb_c
            grad_a = Fa_c - Calpha[kid][:,:nmo_alpha] @ c_Fa_c
            grad_b = Fb_c - Cbeta[kid][:,:nmo_beta] @ c_Fb_c

            error_vector = np.hstack( (error_vector, grad_a.flatten(), grad_b.flatten() ) )

        # norm of the orbital gradient for convergence check
        conv = np.linalg.norm(error_vector)

        solution_vector = np.hstack( (rho_alpha.flatten(), rho_beta.flatten()) )
        new_solution_vector = diis.update(solution_vector, error_vector)

        rho_alpha = new_solution_vector[:len(solution_vector)//2].reshape(rho_alpha_old.shape)
        rho_beta = new_solution_vector[len(solution_vector)//2:].reshape(rho_beta_old.shape)

        rho_alpha = rho_alpha.clip(min = 0).real
        rho_beta = rho_beta.clip(min = 0).real

        # recompute potentials after diis extrapolation
        v_coulomb, v_alpha_r, v_beta_r = compute_local_potentials(rho_alpha, rho_beta, v_ne, basis, xc, libxc_x_functional, libxc_c_functional, jellium)

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        rho_alpha = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
        rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        # diagonalize fock matrix with extrapolated potential
        for kid in range ( len(basis.kpts) ):

            def lobpcg_alpha(c):
                return fock_on_orbitals_using_ace(basis, kid, nalpha, nmo_alpha, phi_alpha, c, T, v_alpha_r, xc, Ki_alpha[kid], B_alpha_ace[kid], ace_exchange, occ_num_alpha, jellium)

            def lobpcg_beta(c):
                return fock_on_orbitals_using_ace(basis, kid, nbeta, nmo_beta, phi_beta, c, T, v_beta_r, xc, Ki_beta[kid], B_beta_ace[kid], ace_exchange, occ_num_beta, jellium)

            Fa_C = LinearOperator((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), matvec=lobpcg_alpha, dtype='complex128')
            Fb_C = LinearOperator((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), matvec=lobpcg_beta, dtype='complex128')

            epsilon_alpha[kid], Calpha[kid] = scipy.sparse.linalg.lobpcg(Fa_C, Calpha[kid], largest=False, maxiter=200, tol=d_convergence*1e-1)
            if not jellium:
                epsilon_beta[kid], Cbeta[kid] = scipy.sparse.linalg.lobpcg(Fb_C, Cbeta[kid], largest=False, maxiter=200, tol=d_convergence*1e-1)
            else:
                Cbeta[kid] = Calpha[kid].copy()
                epsilon_beta[kid] = epsilon_alpha[kid].copy()

            # HF orbitals need to be shifted for smearing to work correctly
            if xc == 'hf':
                epsilon_alpha[kid] -= madelung * occ_num_alpha
                epsilon_beta[kid] -= madelung * occ_num_beta

            # update occupation numbers for smearing
            mu_alpha = find_chemical_potential(nalpha, epsilon_alpha[kid], kBT = kBT)
            mu_beta = find_chemical_potential(nbeta, epsilon_beta[kid], kBT = kBT)

            occ_num_alpha = fermi_dirac(nalpha, epsilon_alpha[kid], mu_alpha, kBT = kBT)
            occ_num_beta = fermi_dirac(nbeta, epsilon_beta[kid], mu_beta, kBT = kBT)

            # break spin symmetry? # TODO this is broken, which probably indicates there is some other problem ...
            #if guess_mix and scf_iter == 0:

            #    c = np.cos(0.05 * np.pi)
            #    s = np.sin(0.05 * np.pi)

            #    tmp1 = c * Calpha[kid][:, nalpha-1] - s * Calpha[kid][:, nalpha]
            #    tmp2 = s * Calpha[kid][:, nalpha-1] + c * Calpha[kid][:, nalpha]

            #    Calpha[kid][:, nalpha-1] = tmp1.copy()
            #    Calpha[kid][:, nalpha] = tmp2.copy()

            # why do i need to orthonormalize my orbitals???
            Calpha[kid] = orthonormalize(Calpha[kid])
            Cbeta[kid] = orthonormalize(Cbeta[kid])
            if jellium:
                Cbeta[kid] = Calpha[kid].copy()

            # update density
            my_rho_alpha, phi_alpha = get_density(basis, Calpha[kid], nalpha, nmo_alpha, kid, occ_num_alpha)
            my_rho_beta, phi_beta = get_density(basis, Cbeta[kid], nbeta, nmo_beta, kid, occ_num_beta)

            # density should be non-negative ...
            rho_alpha += my_rho_alpha.clip(min = 0).real
            rho_beta += my_rho_beta.clip(min = 0).real

            # kinetic energy part of the one-electron energy
            one_electron_energy += np.sum(np.einsum('pi,p->i', np.abs(Calpha[kid][:,:nmo_alpha])**2 * occ_num_alpha, T[kid]))
            one_electron_energy += np.sum(np.einsum('pi,p->i', np.abs(Cbeta[kid][:,:nmo_beta])**2 * occ_num_beta, T[kid]))

            # nonlocal pseudopotential part of the energy
            if not jellium and basis.use_pseudopotential:

                Vnl_c = nonlocal_pseudopotential_on_orbitals(basis, kid, Calpha[kid][:, :nmo_alpha])
                one_electron_energy += np.dot(Calpha[kid][:, :nmo_alpha].conj().flatten(), Vnl_c.flatten())

                Vnl_c = nonlocal_pseudopotential_on_orbitals(basis, kid, Calpha[kid][:, :nmo_alpha])
                one_electron_energy += np.dot(Cbeta[kid][:, :nmo_beta].conj().flatten(), Vnl_c.flatten())

        # nuclear potential / local pseudopotential part of the one-electron energy
        if not jellium:
            v_ne_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
            v_ne_r.ravel()[basis.flat_idx] = v_ne
            v_ne_r = np.fft.fftn(v_ne_r).real
            one_electron_energy += ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum((rho_alpha + rho_beta) * v_ne_r)

        # coulomb part of the energy: 1/2 J
        v_coulomb, v_alpha_r, v_beta_r = compute_local_potentials(rho_alpha, rho_beta, v_ne, basis, xc, libxc_x_functional, libxc_c_functional, jellium)
        v_coulomb_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        v_coulomb_r.ravel()[basis.flat_idx] = v_coulomb
        v_coulomb_r = np.fft.fftn(v_coulomb_r).real
        coulomb_energy = 0.5 * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum((rho_alpha + rho_beta) * v_coulomb_r)

        # save old density
        rho_alpha_old = rho_alpha.copy()
        rho_beta_old = rho_beta.copy()

        # exchange-correlation energy
        if xc != 'hf' :
            xc_energy = get_xc_energy(xc, basis, rho_alpha, rho_beta, libxc_x_functional, libxc_c_functional)

        else :
            # exact exchange energy
            xc_energy = get_exact_exchange_energy(basis, phi_alpha, nmo_alpha, Calpha, occ_num_alpha)
            if nbeta > 0:
                my_xc_energy = get_exact_exchange_energy(basis, phi_beta, nmo_beta, Cbeta, occ_num_beta)
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
        charge = ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.absolute(rho_alpha + rho_beta))

        if jellium:
            if print_level > 0:
                print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf %20.12f %20.12f %20.12f %20.12f %20.12f" %  ( scf_iter, new_total_energy, energy_diff, conv, charge, np.linalg.norm(Calpha[kid] - Cbeta[kid]), np.real(one_electron_energy), np.real(coulomb_energy), np.real(xc_energy), -0.5 * (nalpha + nbeta) * madelung))
        else :
            if print_level > 0:
                print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf" %  ( scf_iter, new_total_energy, energy_diff, conv, charge))

        if conv < d_convergence and energy_diff < e_convergence:
            break

    if scf_iter == maxiter - 1:
        if print_level > 0:
            print('')
            print('    UKS iterations did not converge.')
            print('')
    else:
        if print_level > 0:
            print('')
            print('    UKS iterations converged!')
            print('')

    if print_level > 0:
        print('    ==> energy components <==')
        print('')
        print('    nuclear repulsion energy: %20.12lf' % ( enuc ) )
        print('    one-electron energy:      %20.12lf' % ( np.real(one_electron_energy) ) )
        print('    coulomb energy:           %20.12lf' % ( np.real(coulomb_energy) ) )
        print('    xc energy:                %20.12lf' % ( np.real(xc_energy) ) )
    if jellium:
        if print_level > 0:
            print('    Madelung:                 %20.12lf' % ( -0.5 * (nalpha + nbeta) * madelung) )
    if print_level > 0:
        print('')
    if jellium:
        if print_level > 0:
            print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc -0.5 * (nalpha + nbeta) * madelung ) )
    else :
        if print_level > 0:
            print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc ) )
    if print_level > 0:
        print('')

    return new_total_energy, Calpha, Cbeta

