"""

plane wave scf

"""

import numpy as np
import scipy

from pw_pbc.diis import DIIS

from pw_pbc.pseudopotential import get_local_pseudopotential_gth
from pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements

from pw_pbc.basis import get_miller_indices

from pyscf.pbc import tools

def get_exact_exchange_energy(basis, occupied_orbitals, N):
    """

    evaluate the exact Hartree-Fock exchange energy, according to

        Ex = - 2 pi / Omega sum_{mn in occ} sum_{g} |Cmn(g)|^2 / |g|^2

    where

        Cmn(g) = FT[ phi_m(r) phi_n*(r) ]

    see JCP 108, 4697 (1998) for more details.

    :param basis: plane wave basis information
    :param occupied_orbitals: a list of occupied orbitals
    :param N: the number of electrons

    :return exchange_energy: the exact Hartree-Fock exchange energy
    :return exchange_potential: the exact Hartree-Fock exchange potential

    """


    # accumulate exchange energy and matrix
    exchange_energy = 0.0
    exchange_matrix = np.zeros(len(basis.g), dtype = 'complex128')
    Cmn = np.zeros(len(basis.g), dtype = 'complex128')
    for m in range(0, N):
        for n in range(0, N):

            tmp = occupied_orbitals[m] * occupied_orbitals[n].conj()
            tmp = np.fft.ifftn(tmp)
            for myg in range( len(basis.g) ):
                Cmn[myg] = tmp[ get_miller_indices(myg, basis) ]

            # hmn(g)
            tmp = np.divide(Cmn, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

            exchange_energy += np.sum( Cmn.conj() * tmp )

            # hmn(r) 
            #tmp = np.fft.fftn(tmp) 
            # hmn(g) 
            #tmp = np.fft.ifftn(tmp) 

            exchange_matrix += tmp

    return -2.0 * np.pi / basis.omega * exchange_energy, -4.0 * np.pi / basis.omega * exchange_matrix

def get_xc_potential(xc, basis, rho, occupied_orbitals, N):
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho: the alpha or beta spin density (real space)
    :param occupied_orbitals: list of occupied orbitals (real space)
    :param N: the number of electrons
    :return xc_potential: the exchange-correlation energy

    """

    v_xc = np.zeros(len(basis.g), dtype = 'complex128')

    if xc == 'lda' :

        # LSDA potential
        cx = - 3.0 / 4.0 * ( 3.0 / np.pi )**( 1.0 / 3.0 )
        vr = 4.0 / 3.0 * cx * 2.0 ** ( 1.0 / 3.0 ) * np.power( rho , 1.0 / 3.0 )

        tmp = np.fft.ifftn(vr)
        for myg in range( len(basis.g) ):
            v_xc[myg] = tmp[ get_miller_indices(myg, basis) ]

    elif xc == 'hf' :

        # TODO: fix for general k
        xc_energy, v_xc = get_exact_exchange_energy(basis, occupied_orbitals, N)

    else:
        raise Exception("unsupported xc functional")

    return v_xc

def get_xc_energy(xc, basis, rho, occupied_orbitals, N):
    """

    evaluate the exchange-correlation energy

    :param xc: the exchange-correlation functional name
    :param basis: plane wave basis information
    :param rho: the alpha or beta spin density (real space)
    :param occupied_orbitals: list of occupied orbitals (real space)
    :param N: the number of electrons
    :return xc_potential: the exchange-correlation energy

    """

    xc_energy = 0.0

    if xc == 'lda' :

        # LSDA XC energy
        cx = - 3.0 / 4.0 * ( 3.0 / np.pi )**( 1.0 / 3.0 )
        xc_energy = cx * 2.0 ** ( 1.0 / 3.0 ) * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.power(rho, 4.0/3.0))

    elif xc == 'hf' :

        # TODO: fix for general k
        xc_energy, v_xc = get_exact_exchange_energy(basis, occupied_orbitals, N)

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
        inds = basis.miller_to_g[tuple(gdiff.T.tolist())]

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
    for pp in range(basis.n_plane_waves_per_k[kid]):
        oei[pp][pp] *= 0.5

    diagonal_oei = np.einsum('pi,pq,qj->ij',C.conj(),oei,C)

    one_electron_energy = 0.0
    for pp in range(N):
        one_electron_energy += ( diagonal_oei[pp][pp] ) / len(basis.kpts)

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

    oei = oei + oei.conj().T
    for pp in range(basis.n_plane_waves_per_k[kid]):
        oei[pp][pp] *= 0.5

    diagonal_oei = np.einsum('pi,pq,qj->ij',C.conj(),oei,C)

    coulomb_energy = 0.0
    for pp in range(N):
        coulomb_energy += 0.5 * ( diagonal_oei[pp][pp] ) / len(basis.kpts)

    return coulomb_energy

def uks(cell, basis, xc = 'lda', guess_mix = True):

    """

    plane wave unrestricted kohn-sham

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param xc: the exchange-correlation functional

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

    # maximum number of iterations
    maxiter = 100

    # density in reciprocal space
    rhog = np.zeros(len(basis.g), dtype = 'complex128')

    # density in real space
    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    rho_alpha = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

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

    # damp densities (helps with convergence sometimes)
    damp_densities = True

    # diis 
    diis_dimension = 8
    diis_start_cycle = 2
    diis_update = DIIS(diis_dimension, start_iter = diis_start_cycle)
    old_solution_vector = np.hstack( (rho_alpha.flatten(), rho_beta.flatten()) )

    print("")
    print('    no. k-points:                                %20i' % ( len(basis.kpts) ) )
    print('    KE cutoff (eV)                               %20.2f' % ( basis.ke_cutoff * 27.21138602 ) )
    print('    no. basis functions (orbitals, gamma point): %20i' % ( basis.n_plane_waves_per_k[0] ) )
    print('    no. basis functions (density):               %20i' % ( len(basis.g) ) )
    print('    total_charge:                                %20i' % ( total_charge ) )
    print('    no. alpha bands:                             %20i' % ( nalpha ) )
    print('    no. beta bands:                              %20i' % ( nbeta ) )
    print('    break spin symmetry:                         %20s' % ( "yes" if guess_mix is True else "no" ) )
    print('    damp densities:                              %20s' % ( "yes" if damp_densities is True else "no" ) )
    print('    diis start iteration:                        %20i' % ( diis_start_cycle ) )
    print('    no. diis vectors:                            %20i' % ( diis_dimension ) )

    print("")
    print("    ==> Begin UKS Iterations <==")
    print("")

    print("    %5s %20s %20s %20s %10s" % ('iter', 'energy', '|dE|', '|drho|', 'Nelec'))

    old_total_energy = 0.0

    scf_iter = 0


    # begin UKS iterations
    for i in range(0, maxiter):

        new_rho_alpha = np.zeros(basis.real_space_grid_dim, dtype= 'float64')
        new_rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

        occ_alpha = []
        occ_beta = []

        one_electron_energy = 0.0
        coulomb_energy = 0.0

        va = v_coulomb + v_xc_alpha + v_ne
        vb = v_coulomb + v_xc_beta + v_ne

        # loop over k-points
        for kid in range( len(basis.kpts) ):

            # alpha

            # form fock matrix
            fock = form_fock_matrix(basis, kid, v = va)

            # diagonalize fock matrix
            n = nalpha - 1
            if scf_iter == 0 and guess_mix == True :
                n = nalpha
            epsilon_alpha, Calpha = scipy.linalg.eigh(fock, lower = False, eigvals=(0,n))
            #print(epsilon_alpha - madelung)
            
            # break spin symmetry?
            if guess_mix is True and scf_iter == 0:

                c = np.cos(0.25 * np.pi)
                s = np.sin(0.25 * np.pi)

                tmp1 = c * Calpha[:, nalpha-1] - s * Calpha[:, nalpha]
                tmp2 = s * Calpha[:, nalpha-1] + c * Calpha[:, nalpha]

                Calpha[:, nalpha-1] = tmp1
                Calpha[:, nalpha] = tmp2

            # one-electron part of the energy 
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Calpha, 
                                                           nalpha, 
                                                           kid, 
                                                           v_ne = v_ne)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Calpha, nalpha, kid, v_coulomb)

            # accumulate density 
            rho, occ_alpha = get_density(basis, Calpha, nalpha, kid)
            new_rho_alpha += rho

            # now beta
            if nbeta == 0 : 
                continue

            # form fock matrix
            fock = form_fock_matrix(basis, kid, v = vb)

            # diagonalize fock matrix
            epsilon_beta, Cbeta = scipy.linalg.eigh(fock, lower = False, eigvals=(0,nbeta-1))

            # one-electron part of the energy 
            one_electron_energy += get_one_electron_energy(basis, 
                                                           Cbeta, 
                                                           nbeta, 
                                                           kid, 
                                                           v_ne = v_ne)

            # coulomb part of the energy: 1/2 J
            coulomb_energy += get_coulomb_energy(basis, Cbeta, nbeta, kid, v_coulomb)

            # accumulate density 
            rho, occ_beta = get_density(basis, Cbeta, nbeta, kid)
            new_rho_beta += rho

        xc_energy = get_xc_energy(xc, basis, new_rho_alpha, occ_alpha, nalpha)
        if nbeta > 0:
            xc_energy += get_xc_energy(xc, basis, new_rho_beta, occ_beta, nbeta)

        if xc == 'hf':
            xc_energy -= 0.5 * (nalpha + nbeta) * madelung

        # damp density
        factor = 1.0
        if damp_densities is True and scf_iter > 0 :
           factor = 0.5
        new_rho_alpha = factor * new_rho_alpha + (1.0 - factor) * rho_alpha
        new_rho_beta = factor * new_rho_beta + (1.0 - factor) * rho_beta

        # diis extrapolation
        rho_dim = basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2]
        solution_vector = np.hstack( (new_rho_alpha.flatten(), new_rho_beta.flatten()) )
        error_vector = old_solution_vector - solution_vector
        new_solution_vector = diis_update.compute_new_vec(solution_vector, error_vector)
        new_rho_alpha = new_solution_vector[:rho_dim].reshape(rho_alpha.shape)
        new_rho_beta = new_solution_vector[rho_dim:].reshape(rho_beta.shape)
        old_solution_vector = np.copy(new_solution_vector)

        # convergence in density
        rho_diff_norm = np.linalg.norm(error_vector) 

        # update density
        rho_alpha = new_rho_alpha
        rho_beta = new_rho_beta

        # density should be non-negative ...
        rho_alpha = rho_alpha.clip(min = 0)
        rho_beta = rho_beta.clip(min = 0)

        rho = new_rho_alpha + new_rho_beta

        # coulomb potential
        tmp = np.fft.ifftn(rho)
        for myg in range( len(basis.g) ):
            rhog[myg] = tmp[ get_miller_indices(myg, basis) ]

        v_coulomb = 4.0 * np.pi * np.divide(rhog, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0) # / omega

        # exchange-correlation potential
        v_xc_alpha = get_xc_potential(xc, basis, rho_alpha, occ_alpha, nalpha)

        if nbeta > 0:
            v_xc_beta = get_xc_potential(xc, basis, rho_beta, occ_beta, nbeta)

        # total energy
        new_total_energy = np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc

        # convergence in energy
        energy_diff = np.abs(new_total_energy - old_total_energy)

        # update energy
        old_total_energy = new_total_energy

        # charge
        charge = ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.absolute(rho))

        print("    %5i %20.12lf %20.12lf %20.12lf %10.6lf" %  ( scf_iter, new_total_energy, energy_diff, rho_diff_norm, charge ) )

        if ( rho_diff_norm < 1e-4 and energy_diff < 1e-5 ) :
            break

        scf_iter += 1

    if scf_iter == maxiter:
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

    assert(np.isclose( np.real(one_electron_energy) + np.real(coulomb_energy) + np.real(xc_energy) + enuc, -9.802901383306) )

