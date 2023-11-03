"""

plane wave scf

"""

# libxc
import pylibxc

# TODO: this thing shouldn't be global, and i should be able to select the functional
libxc_functional = pylibxc.LibXCFunctional("lda_x", "polarized")

import numpy as np
import scipy

from qcpanop.pw_pbc.pseudopotential import get_local_pseudopotential_gth
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements

from qcpanop.pw_pbc.basis import get_miller_indices

from pyscf.pbc import tools

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
    :return exchange_potential: the exact Hartree-Fock exchange potential

    """


    # accumulate exchange energy and matrix
    exchange_energy = 0.0

    Cij = np.zeros(len(basis.g), dtype = 'complex128')

    mat = np.zeros((N,N), dtype = 'complex128')

    for i in range(0, N):

        # Ki(r) = sum_j Kij(r) phi_j(r)
        Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            tmp = occupied_orbitals[j].conj() * occupied_orbitals[i]

            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(tmp)
            for myg in range( len(basis.g) ):
                Cij[myg] = tmp[ get_miller_indices(myg, basis) ]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij = np.divide(Cij, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

            exchange_energy += np.sum( Cij.conj() * Kij )

    #        # Kij(r) = FFT^-1[Kij(g)]
    #        Kij_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
    #        for myg in range( len(basis.g) ):
    #            Kij_r[ get_miller_indices(myg, basis) ] = Kij[myg]
    #        Kij_r = np.fft.fftn(Kij_r)

    #        # action of K on an occupied orbital, i: 
    #        # Ki(r) = sum_j Kij(r) phi_j(r)
    #        Ki_r += Kij_r * occupied_orbitals[j]

    #    # build exchange matrix in MO basis <phi_k | K | phi_i>
    #    factor = ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) )
    #    for k in range(0, N):
    #        mat[k, i] = factor * np.sum( Ki_r * occupied_orbitals[k].conj() ) 

    ## transform exchange matrix back from MO basis
    #trans = np.zeros((basis.n_plane_waves_per_k[0], N), dtype = 'complex128')
    #for i in range(0, N):
    #    trans[:, i] = C[:, i]
    #    
    #exchange_matrix = np.einsum('pi,ij,qj->pq', trans.conj(), mat, trans)


    # try new way ...
    exchange_matrix = np.zeros((basis.n_plane_waves_per_k[0], basis.n_plane_waves_per_k[0]), dtype='complex128')

    for i in range(0, basis.n_plane_waves_per_k[0]):
        ii = basis.kg_to_g[0][i]

        # Ki(r) = sum_j Kij(r) phi_j(r)
        Ki_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')

        # a planewave basis function
        phi_i = np.zeros(len(basis.g), dtype = 'complex128')
        phi_i[ii] = 1.0
        tmp = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
        for myg in range( len(basis.g) ):
            tmp[ get_miller_indices(myg, basis) ] = phi_i[myg]
        phi_i = np.fft.fftn(tmp)

        for j in range(0, N):

            # Cij(r') = phi_i(r') phi_j*(r')
            tmp = occupied_orbitals[j].conj() * phi_i #occupied_orbitals[i]

            # Cij(g) = FFT[Cij(r')]
            tmp = np.fft.ifftn(tmp)
            for myg in range( len(basis.g) ):
                Cij[myg] = tmp[ get_miller_indices(myg, basis) ]

            # Kij(g) = Cij(g) * FFT[1/|r-r'|]
            Kij = np.divide(Cij, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

            # Kij(r) = FFT^-1[Kij(g)]
            Kij_r = np.zeros(basis.real_space_grid_dim, dtype = 'complex128')
            for myg in range( len(basis.g) ):
                Kij_r[ get_miller_indices(myg, basis) ] = Kij[myg]
            Kij_r = np.fft.fftn(Kij_r)

            # action of K on an occupied orbital, i: 
            # Ki(r) = sum_j Kij(r) phi_j(r)
            Ki_r += Kij_r * occupied_orbitals[j]

        # build a row of the exchange matrix: < G' | K | G''> = FFT( Ki_r )
        row_g = np.fft.ifftn(Ki_r)
        for k in range(0, basis.n_plane_waves_per_k[0]):
            kk = basis.kg_to_g[0][k]
            
            exchange_matrix[k, i] = row_g[ get_miller_indices(kk, basis) ] 

    return -2.0 * np.pi / basis.omega * exchange_energy, -4.0 * np.pi / basis.omega * exchange_matrix

def get_xc_potential(xc, basis, rho_alpha, rho_beta):
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho_alpha: the alpha spin density (real space)
    :param rho_beta: the beta spin density (real space)
    :return xc_alpha: the exchange-correlation energy (alpha)
    :return xc_beta: the exchange-correlation energy (beta)

    """


    v_xc_alpha = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_beta = np.zeros(len(basis.g), dtype = 'complex128')

    if xc == 'lda' :

        ## LSDA potential 
        #cx = - 3.0 / 4.0 * ( 3.0 / np.pi )**( 1.0 / 3.0 )
        #vr = 4.0 / 3.0 * cx * 2.0 ** ( 1.0 / 3.0 ) * np.power( rho , 1.0 / 3.0 )

        #tmp = np.fft.ifftn(vr)
        #for myg in range( len(basis.g) ):
        #    v_xc[myg] = tmp[ get_miller_indices(myg, basis) ]

        # libxc wants a list of density elements [alpha[0], beta[0], alpha[1], beta[1], etc.]
        combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
        combined_rho[::2] = rho_alpha.ravel(order='C')
        combined_rho[1::2] = rho_beta.ravel(order='C')

        # compute
        ret = libxc_functional.compute( combined_rho )

        # unpack v_xc(r) and fourier transform
        vrho = ret['vrho']
        tmp_alpha = vrho[:, 0].reshape(basis.real_space_grid_dim)
        tmp_beta = vrho[:, 1].reshape(basis.real_space_grid_dim)
        tmp_alpha = np.fft.ifftn(tmp_alpha)
        tmp_beta = np.fft.ifftn(tmp_beta)

        # unpack v_xc(g) 
        for myg in range( len(basis.g) ):
            v_xc_alpha[myg] = tmp_alpha[ get_miller_indices(myg, basis) ]
            v_xc_beta[myg] = tmp_beta[ get_miller_indices(myg, basis) ]

    elif xc == 'hf' :

        pass

        # TODO: fix for general k
        #xc_energy, v_xc = get_exact_exchange_energy(basis, occupied_orbitals, N, C)

    else:
        raise Exception("unsupported xc functional")

    return v_xc_alpha, v_xc_beta

def get_xc_energy(xc, basis, rho_alpha, rho_beta): 
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho_alpha: the alpha spin density (real space)
    :param rho_beta: the beta spin density (real space)

    """

    xc_energy = 0.0

    if xc == 'lda':

        # libxc wants a list of density elements [alpha[0], beta[0], alpha[1], beta[1], etc.]
        combined_rho = np.zeros((2 * np.prod(basis.real_space_grid_dim[:3])))
        combined_rho[::2] = rho_alpha.ravel(order='C')
        combined_rho[1::2] = rho_beta.ravel(order='C')

        # compute
        ret = libxc_functional.compute( combined_rho , do_vxc = False)
        zk = ret['zk']
        val = (rho_alpha.flatten() + rho_beta.flatten() ) * zk.flatten()
        xc_energy = val.sum() * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) )

    else:
        raise Exception("unsupported xc functional")

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

def get_density(basis, C, N, kid):
    """

    get real-space density from molecular orbital coefficients

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param N: the number of electrons
    :param kid: index for a given k-point

    :return rho: the density

    """

    occupied_orbitals = []
    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    for pp in range(N):

        occ = np.zeros(basis.real_space_grid_dim,dtype = 'complex128')

        for tt in range( basis.n_plane_waves_per_k[kid] ):

            ik = basis.kg_to_g[kid][tt]
            occ[ get_miller_indices(ik, basis) ] = C[tt, pp]

        occ = np.fft.fftn(occ) 
        occupied_orbitals.append( occ )

        rho += np.absolute(occ)**2.0 / basis.omega

    return ( 1.0 / len(basis.kpts) ) * rho, occupied_orbitals

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
    
    # get non-local part of the pseudopotential
    if basis.use_pseudopotential:
        fock += get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)

    # get kinetic energy
    kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]
    diagonals = np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0 + fock.diagonal()
    np.fill_diagonal(fock, diagonals)

    # only upper triangle of fock is populated ... symmetrize and rescale diagonal
    fock = fock + fock.conj().T
    diag = np.diag(fock)
    np.fill_diagonal(fock, 0.5 * diag)

    return fock

def get_one_electron_energy(basis, C, N, kid, v_ne = None):
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

    if v_ne is not None:
        oei += get_matrix_elements(basis, kid, v_ne)

    if basis.use_pseudopotential:
        oei += get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = basis.nl_pp_use_legendre)

    kgtmp = basis.kpts[kid] + basis.g[basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]]

    diagonals = np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0 + oei.diagonal()
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

def uks(cell, basis, xc = 'lda', guess_mix = True, diis_dimension = 8, damp_fock = True, damping_iterations = 8):

    """

    plane wave unrestricted kohn-sham

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param xc: the exchange-correlation functional
    :param guess_mix: do mix alpha homo and lumo to break spin symmetry?
    :param damp_fock: do dampen fock matrix?
    :param damping_iterations: for how many iterations should we dampen the fock matrix

    """
 
    print('')
    print('    ************************************************')
    print('    *                                              *')
    print('    *                Plane-wave UKS                *')
    print('    *                                              *')
    print('    ************************************************')
    print('')

    if xc != 'lda' and xc != 'hf':
        raise Exception("uks only supports xc = 'lda' and 'hf' for now")


    # get nuclear repulsion energy
    enuc = cell.energy_nuc()

    # coulomb and xc potentials in reciprocal space
    v_coulomb = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_alpha = np.zeros(len(basis.g), dtype = 'complex128')
    v_xc_beta = np.zeros(len(basis.g), dtype = 'complex128')

    exchange_matrix_alpha = np.zeros((basis.n_plane_waves_per_k[0], basis.n_plane_waves_per_k[0]), dtype='complex128')
    exchange_matrix_beta = np.zeros((basis.n_plane_waves_per_k[0], basis.n_plane_waves_per_k[0]), dtype='complex128')

    # maximum number of iterations
    maxiter = 500

    # density in reciprocal space
    rhog = np.zeros(len(basis.g), dtype = 'float64')

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

    # number of alpha and beta bands
    total_charge = 0
    for I in range ( len(valence_charges) ):
        total_charge += valence_charges[I]

    total_charge -= basis.charge

    nbeta = int(total_charge / 2)
    nalpha = total_charge - nbeta

    # damp fock matrix (helps with convergence sometimes)
    damping_factor = 1.0
    if damp_fock :
        damping_factor = 0.5

    # diis 
    diis_dimension = 8
    diis_start_cycle = 4

    print("")
    print('    no. k-points:                                %20i' % ( len(basis.kpts) ) )
    print('    KE cutoff (eV)                               %20.2f' % ( basis.ke_cutoff * 27.21138602 ) )
    print('    no. basis functions (orbitals, gamma point): %20i' % ( basis.n_plane_waves_per_k[0] ) )
    print('    no. basis functions (density):               %20i' % ( len(basis.g) ) )
    print('    total_charge:                                %20i' % ( total_charge ) )
    print('    no. alpha bands:                             %20i' % ( nalpha ) )
    print('    no. beta bands:                              %20i' % ( nbeta ) )
    print('    break spin symmetry:                         %20s' % ( "yes" if guess_mix is True else "no" ) )
    print('    damp fock matrix:                            %20s' % ( "yes" if damp_fock is True else "no" ) )
    print('    no. damping iterations:                      %20i' % ( damping_iterations ) )
    #print('    diis start iteration:                        %20i' % ( diis_start_cycle ) )
    print('    no. diis vectors:                            %20i' % ( diis_dimension ) )

    print("")
    print("    ==> Begin UKS Iterations <==")
    print("")

    print("    %5s %20s %20s %20s %10s" % ('iter', 'energy', '|dE|', '||[F, D]||', 'Nelec'))

    old_total_energy = 0.0

    fock_a = []
    fock_b = []

    Calpha = []
    Cbeta = []

    old_fock_a = []
    old_fock_b = []

    for kid in range ( len(basis.kpts) ):

        fock_a.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))
        fock_b.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))

        old_fock_a.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))
        old_fock_b.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))

        Calpha.append(np.eye(basis.n_plane_waves_per_k[kid]))
        Cbeta.append(np.eye(basis.n_plane_waves_per_k[kid]))

    from pyscf import lib
    adiis = lib.diis.DIIS()
    adiis.space = diis_dimension

    # begin UKS iterations
    for scf_iter in range(0, maxiter):

        occ_alpha = []
        occ_beta = []

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        if xc == 'lda' :
            va = v_coulomb + v_ne + v_xc_alpha
            vb = v_coulomb + v_ne + v_xc_beta
        elif xc == 'hf' :
            va = v_coulomb + v_ne 
            vb = v_coulomb + v_ne 
        else:
            raise Exception("unsupported xc functional")

        # zero density for this iteration
        rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        # form each block of the fock matrix and orbital gradient
        damp = 1.0
        if scf_iter > 0 and scf_iter < 1 * diis_start_cycle :
            damp = damping_factor

        # form fock matrix and orbital gradient

        fock_a = []
        fock_b = []
        grad_a = []
        grad_b = []

        for kid in range ( len(basis.kpts) ):
            fock_a.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))
            fock_b.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))
            grad_a.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))
            grad_b.append(np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype = 'complex128'))

        # damping factor
        damp = 1.0
        if scf_iter < diis_start_cycle and damp_fock : 
            damp = damping_factor

        # loop over k-points
        for kid in range( len(basis.kpts) ):

            # form fock matrix
            fock_a[kid] = form_fock_matrix(basis, kid, v = va)
            fock_b[kid] = form_fock_matrix(basis, kid, v = vb)

            # exact exchange?
            if xc == 'hf' :
                fock_a += exchange_matrix_alpha
                fock_b += exchange_matrix_beta

            # damp fock matrix
            fock_a[kid] = damp * fock_a[kid] + (1.0 - damp) * old_fock_a[kid]
            fock_b[kid] = damp * fock_b[kid] + (1.0 - damp) * old_fock_b[kid]
            old_fock_a[kid] = fock_a[kid].copy()
            old_fock_b[kid] = fock_b[kid].copy()

            # form opdm and orbital gradient (for diis)
            grad_a[kid] = form_orbital_gradient(basis, Calpha[kid], nalpha, fock_a[kid], kid)
            grad_b[kid] = form_orbital_gradient(basis, Cbeta[kid], nbeta, fock_b[kid], kid)

        # extrapolate fock matrix

        # solution vector is fock matrix
        solution_vector = np.zeros(0)
        for kid in range ( len(basis.kpts) ):
            solution_vector = np.hstack( (solution_vector, fock_a[kid].flatten(), fock_b[kid].flatten() ) )

        # error vector is orbital gradient
        error_vector = np.zeros(0)
        for kid in range ( len(basis.kpts) ):
            error_vector = np.hstack( (error_vector, grad_a[kid].flatten(), grad_b[kid].flatten() ) )

        # norm of orbital gradient
        conv = np.linalg.norm(error_vector)

        # extrapolate solution vector
        new_solution_vector = adiis.update(solution_vector, error_vector)

        # reshape solution vector
        off = 0
        for kid in range ( len(basis.kpts) ):
            dim = basis.n_plane_waves_per_k[kid]
            fock_a[kid] = new_solution_vector[off:off+dim*dim].reshape(fock_a[kid].shape)
            off += dim*dim
            fock_b[kid] = new_solution_vector[off:off+dim*dim].reshape(fock_b[kid].shape)
            off += dim*dim

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        rho_a = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
        rho_b = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        # diagonalize extrapolated fock matrix
        for kid in range ( len(basis.kpts) ):

            n = nalpha - 1
            if scf_iter == 0 and guess_mix == True :
                n = nalpha

            epsilon_alpha, Calpha[kid] = scipy.linalg.eigh(fock_a[kid], eigvals=(0, n))
            epsilon_beta, Cbeta[kid] = scipy.linalg.eigh(fock_b[kid], eigvals=(0, (nbeta-1)))

            #epsilon_alpha, Calpha = np.linalg.eigh(fock_a)
            #epsilon_beta, Cbeta = np.linalg.eigh(fock_b)

            # break spin symmetry?
            if guess_mix is True and scf_iter == 0:

                c = np.cos(0.25 * np.pi)
                s = np.sin(0.25 * np.pi)

                tmp1 = c * Calpha[kid][:, nalpha-1] - s * Calpha[kid][:, nalpha]
                tmp2 = s * Calpha[kid][:, nalpha-1] + c * Calpha[kid][:, nalpha]

                Calpha[kid][:, nalpha-1] = tmp1
                Calpha[kid][:, nalpha] = tmp2

            # update density
            my_rho_a, occ_alpha = get_density(basis, Calpha[kid], nalpha, kid)
            my_rho_b, occ_beta = get_density(basis, Cbeta[kid], nbeta, kid)

            # density should be non-negative ...
            rho_a += my_rho_a.clip(min = 0)
            rho_b += my_rho_b.clip(min = 0)

            # one-electron part of the energy (alpha)
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Calpha[kid], 
                                                           nalpha, 
                                                           kid, 
                                                           v_ne = v_ne)

            # one-electron part of the energy (beta)
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Cbeta[kid], 
                                                           nbeta, 
                                                           kid, 
                                                           v_ne = v_ne)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Calpha[kid], nalpha, kid, v_coulomb)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Cbeta[kid], nbeta, kid, v_coulomb)

        rho = rho_a + rho_b


        if xc == 'lda':

            xc_energy = get_xc_energy(xc, basis, rho_a, rho_b)

        if xc == 'hf':

            # TODO: fix for general k
            xc_energy, v_xc = get_exact_exchange_energy(basis, occ_alpha, nalpha, Calpha)
            if nbeta > 0:
                my_xc_energy, v_xc = get_exact_exchange_energy(basis, occ_beta, nbeta, Cbeta)
                xc_energy += my_xc_energy

            xc_energy -= 0.5 * (nalpha + nbeta) * madelung

        # coulomb potential
        tmp = np.fft.ifftn(rho)
        for myg in range( len(basis.g) ):
            rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

        v_coulomb = 4.0 * np.pi * np.divide(rhog, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0) # / omega

        # exchange-correlation potential
        if xc == 'lda' :

            v_xc_alpha, v_xc_beta = get_xc_potential(xc, basis, rho_a, rho_b)

        elif xc == 'hf' :

            dum, exchange_matrix_alpha = get_exact_exchange_energy(basis, occ_alpha, nalpha, Calpha)
            if nbeta > 0:
                dum, exchange_matrix_beta = get_exact_exchange_energy(basis, occ_beta, nbeta, Cbeta)

        else:
            raise Exception("unsupported xc functional")

        # total energy
        new_total_energy = np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc

        # convergence in energy
        energy_diff = np.abs(new_total_energy - old_total_energy)

        # update energy
        old_total_energy = new_total_energy

        # charge
        charge = ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.absolute(rho))

        print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf" %  ( scf_iter, new_total_energy, energy_diff, conv, charge ) )

        if ( conv < 1e-4 and energy_diff < 1e-5 ) :
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
    print('')
    print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc ) )
    print('')

    #assert(np.isclose( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc, -9.802901383306) )

    return np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc
