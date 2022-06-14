"""

Use PySCF infrastructure to perform SCF with plane wave basis

"""
from itertools import product
import warnings
import numpy as np
from pyscf.pbc import gto, scf
# divide by this quantity to get value in Bohr (atomic unit length)
from pyscf.lib.parameters import BOHR  # angstrom / bohr
from pyscf import lib
from pyscf.pbc.gto import estimate_ke_cutoff
from pyscf.pbc import tools

import ase
from ase.build import bulk
from pyscf.pbc import gto as pbcgto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import scipy

# GTH pseudopotential parameters
class gth_pseudopotential_parameters():

    # parameters for local contribution to pseudopotential (default Si)

    c1 = -7.336103
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0
    rloc = 0.44
    Zion = 4.0

    # parameters for non-local contribution to pseudopotential (default Si)

    # hlij
    hgth = np.zeros((2,3,3),dtype='float64')
    hgth[0, 0, 0] = 5.906928
    hgth[0, 1, 1] = 3.258196
    hgth[0, 0, 1] = -0.5*np.sqrt(3./5.) * hgth[0,1,1]
    hgth[0, 1, 0] = -0.5*np.sqrt(3./5.) * hgth[0,1,1]
    hgth[1, 0, 0] = 2.727013

    # rs, rp, max l, max i
    r0 = 0.422738
    r1 = 0.484278
    rl = [r0, r1]
    lmax = 2
    imax = 2

def get_local_pseudopotential_gth(g2, omega, gth_params, tiny = 1e-8):

    """

    Construct and return local contribution to GTH pseudopotential

    :param g2: square modulus of plane wave basis functions
    :param omega: unit cell volume
    :param gth_params: GTH pseudopotential parameters
    :return: local contribution to GTH pseudopotential
    """

    c1 = gth_params.c1
    c2 = gth_params.c2
    c3 = gth_params.c3
    c4 = gth_params.c4
    rloc = gth_params.rloc
    Zion = gth_params.Zion

    vsg = np.zeros(len(g2),dtype='float64')
    largeind = g2 > tiny
    smallind = g2 <= tiny #|G|^2->0 limit

    g2 = g2[largeind]

    rloc2 = rloc * rloc
    rloc3 = rloc * rloc2
    rloc4 = rloc2 * rloc2
    rloc6 = rloc2 * rloc4

    g4 = g2 * g2
    g6 = g2 * g4

    vsgl=np.exp(-g2*rloc2/2.)*(-4.*np.pi*Zion/g2+np.sqrt(8.*np.pi**3.)*rloc3*(c1+c2*(3.-g2*rloc2)+c3*(15.-10.*g2*rloc2+g4*rloc4)+c4*(105.-105.*g2*rloc2+21.*g4*rloc4-g6*rloc6)))
    vsgs=2.*np.pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*np.pi)*rloc-Zion) #|G|^2->0 limit ... AED changed sign

    vsg[largeind] = vsgl
    vsg[smallind] = vsgs

    return vsg / omega

def get_spherical_harmonics_and_projectors_gth(gv, gth_params):

    """

    Construct spherical harmonics and projectors for GTH pseudopotential

    :param gv: plane wave basis functions plus kpt
    :param gth_params: GTH pseudopotential parameters
    :param rl: list of [rs, rp]
    :param lmax: maximum angular momentum, l
    :param imax: maximum i for projectors
    :return: spherical harmonics and projectors for GTH pseudopotential
    """

    rl = gth_params.rl
    lmax = gth_params.lmax
    imax = gth_params.imax

    rgv,thetagv,phigv=pbcgto.pseudo.pp.cart2polar(gv)

    mmax = 2 * (lmax - 1) + 1
    gmax = len(gv)

    spherical_harmonics_lm = np.zeros((lmax,mmax,gmax),dtype='complex128')
    projector_li           = np.zeros((lmax,imax,gmax),dtype='complex128')

    for l in range(lmax):
        for m in range(-l,l+1):
            spherical_harmonics_lm[l,m+l,:] = scipy.special.sph_harm(m,l,phigv,thetagv)
        for i in range(imax):
            projector_li[l,i,:] = pbcgto.pseudo.pp.projG_li(rgv,l,i,rl[l])

    return spherical_harmonics_lm, projector_li

def get_nonlocal_pseudopotential_gth(sphg, pg, gind, gth_params, omega):

    """

    Construct and return non-local contribution to GTH pseudopotential

    :param sphg: angular part of plane wave basis
    :param pg: projectors 
    :param gind: plane wave basis function label 
    :param gth_params: GTH pseudopotential parameters
    :param omega: unit cell volume
    :return: non-local contribution to GTH pseudopotential
    """

    hgth = gth_params.hgth

    vsg = 0.0
    for l in [0,1]:
        vsgij = vsgsp = 0.0
        for i in [0,1]:
            for j in [0,1]:
                #vsgij+=thepow[l]*pg[l,i,gind]*hgth[l,i,j]*pg[l,j,:]
                vsgij += pg[l,i,gind] * hgth[l,i,j] * pg[l,j,:]

        for m in range(-l,l+1):
            vsgsp += sphg[l,m+l,gind] * sphg[l,m+l,:].conj()

        vsg += vsgij * vsgsp

    return vsg / omega


def get_gth_pseudopotential(basis, gth_params, k, kid, omega):

    """

    get the GTH pseudopotential matrix

    :param basis: plane wave basis information
    :param gth_params: GTH pseudopotential parameters
    :param k: the list of k-points
    :param kid: index for a given k-point
    :param omega: the cell volume
    :return gth_pseudopotential: the GTH pseudopotential matrix

    """

    gth_pseudopotential = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype='complex128')

    gkind = basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]
    gk = basis.g[gkind]

    sphg, pg = get_spherical_harmonics_and_projectors_gth(k[kid]+gk,gth_params)

    for aa in range(basis.n_plane_waves_per_k[kid]):

        ik = basis.kg_to_g[kid][aa]
        gdiff = basis.miller[ik] - basis.miller[gkind[aa:]] + np.array(basis.reciprocal_max_dim)
        #inds = basis.miller_to_g[gdiff.T.tolist()]
        inds = basis.miller_to_g[tuple(gdiff.T.tolist())]

        vsg_local = get_local_pseudopotential_gth(basis.g2[inds], omega, gth_params)

        vsg_nonlocal = get_nonlocal_pseudopotential_gth(sphg, pg, aa, gth_params, omega)[aa:]

        gth_pseudopotential[aa, aa:] = 0.0
        for I in range(0, len(basis.SI)):
            gth_pseudopotential[aa, aa:] += ( vsg_local + vsg_nonlocal ) * basis.SI[I][inds]

    return gth_pseudopotential

def get_potential(basis, k, kid, vg):

    """

    get the potential matrix for given k-point

    :param basis: plane wave basis information
    :param k: the list of k-points
    :param kid: index for a given k-point
    :param vg: full potential container
    :return potential: the pseudopotential matrix

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


def get_SI(cell, Gv=None):

    """
    Calculate the structure factor for all atoms; see MH (3.34).

    Args:
        cell : instance of :clast:`Cell`

        Gv : (N,3) array
            G vectors

    Returns:
        SI : (natm, ngs) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    """

    if Gv is None:
        Gv = cell.get_Gv()
    coords = cell.atom_coords()
    SI = np.exp(-1j*np.dot(coords, Gv.T))
    return SI

def get_nuclear_electronic_potential(cell, basis, omega, valence_charges = None):

    """
    v(r) = \sum_{G}v(G)exp(iG.r)

    :param cell: the unit cell
    :param basis: plane wave basis information
    :param valence_charges: the charges associated with the valence
    :return: vneG: the nuclear-electron potential in the plane wave basis
    """

    # nuclear charge of atoms in cell. 
    charges = - cell.atom_charges()
    if valence_charges is not None:
        charges = - valence_charges

    print(charges)

    # structure factor
    SI = get_SI(cell, basis.g)

    assert SI.shape[1] == basis.g.shape[0]

    # this is Z_{I} * S_{I}
    rhoG = np.dot(charges, SI)  
    #for I in range(0, 1):
    #    rhoG -= np.dot(charges[I], basis.SI[I])

    coulG = np.divide(4.0 * np.pi, basis.g2, out=np.zeros_like(basis.g2), where=basis.g2 != 0.0) # save divide

    vneG = rhoG * coulG / omega

    return vneG

def get_cell_info(n_k_points, atom_type, unit_type, lattice_constant, energy_cutoff, cell_dimension = 3, distance_units = 'B'):

    """
       
    get k-points and cell info
    
    :param n_k_points: number of k-points in each direction
    :param atom_type: atom type
    :param unit_type: unit cell type
    :param lattice_constant: lattice constant
    :param energy_cutoff: energy cutoff for plane wave basis functions
    :param cell_dimension: dimensionality (default = 3)
    :param distance_units: distance units (default = 'B', bohr)
    :return k: k-points
    :return a: lattice vectors
    :return h: reciprocal lattice vectors
    :return cell: the unit cell
    :return omega: cell volume

    """

    cell = gto.M(a = np.eye(3) * 5,
                 atom = 'H 0 0 1.0; H 0 0 0',
                 basis = 'sto-3g',
                 unit = 'angstrom',
                 ke_cutoff = 0.5)

    # build unit cell
    #ase_atom = bulk(atom_type, unit_type, a = lattice_constant)

    # get PySCF unit cell object
    #cell = pbcgto.Cell()

    # set atoms
    #cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)

    # set lattice vectors
    #cell.a = ase_atom.cell 

    # set kinetic energy cutoff
    cell.ke_cutoff = energy_cutoff 

    # set precision
    cell.precision = 1.e-8

    # set dimension
    cell.dimension = cell_dimension

    # set units
    cell.unit = distance_units

    # build 
    cell.build()

    # get k-points
    k = cell.make_kpts(n_k_points, wrap_around=True)

    # get lattice vectors
    a = cell.a 

    # get reciprocal lattice vectors
    b = cell.reciprocal_vectors() 

    # get cell volume
    omega = np.linalg.det(a)

    return k, a, b, omega, cell

def factor_integer(n):

    i = 2
    factors=[]
    while i*i <= n:

        if n % i:
            i += 1
        else:
            n //= i
            if i not in factors:
                factors.append(i)

    if n > 1:
        if n not in factors:
            factors.append(n)

    return factors

def get_plane_wave_basis(energy_cutoff, a, b):

    """
    
    get plane wave basis functions and indices

    :param energy_cuttoff: kinetic energy cutoff (in atomic units)
    :param a: lattice vectors
    :param b: reciprocal lattice vectors

    :return g: the plane wave basis
    :return g2: the square modulus of the plane wave basis
    :return miller: miller indices
    :return reciprocal_max_dim: maximum dimensions for reciprocal basis
    :return real_space_grid_dim: real-space grid dimensions
    :return miller_to_g: maps the miller indices to the index of G 

    
    """

    # estimate maximum values for the miller indices (for density)
    reciprocal_max_dim_1 = int( np.ceil( (np.sqrt(2.0*4.0*energy_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[0]) + 1.0))
    reciprocal_max_dim_2 = int( np.ceil( (np.sqrt(2.0*4.0*energy_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[1]) + 1.0))
    reciprocal_max_dim_3 = int( np.ceil( (np.sqrt(2.0*4.0*energy_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[2]) + 1.0))

    # g, g2, and miller indices
    g = np.empty(shape=[0,3],dtype='float64')
    g2 = np.empty(shape=[0,0],dtype='float64')
    miller = np.empty(shape=[0,3],dtype='int')

    for i in np.arange(-reciprocal_max_dim_1, reciprocal_max_dim_1+1):
        for j in np.arange(-reciprocal_max_dim_2, reciprocal_max_dim_2+1):
            for k in np.arange(-reciprocal_max_dim_3, reciprocal_max_dim_3+1):

                # G vector
                gtmp = i * b[0] + j * b[1] + k * b[2]

                # |G|^2
                g2tmp = np.dot(gtmp, gtmp)

                # energy_cutoff for density is 4 times energy_cutoff for orbitals
                if (g2tmp/2.0 <= 4.0 * energy_cutoff):

                    # collect G vectors
                    g = np.concatenate( (g, np.expand_dims(gtmp,axis=0) ) ) 

                    # |G|^2
                    g2 = np.append(g2, g2tmp) 

                    # list of miller indices for G vectors
                    miller = np.concatenate( (miller, np.expand_dims(np.array([i, j, k]), axis = 0) ) ) 

    # reciprocal_max_dim contains the maximum dimension for reciprocal basis
    reciprocal_max_dim = [int(np.amax(miller[:, 0])), int(np.amax(miller[:, 1])), int(np.amax(miller[:, 2]))]

    # real_space_grid_dim is the real space grid dimensions
    real_space_grid_dim = [2 * reciprocal_max_dim[0] + 1, 2 * reciprocal_max_dim[1] + 1, 2 * reciprocal_max_dim[2] + 1]

    # miller_to_g maps the miller indices to the index of G
    miller_to_g = np.ones(real_space_grid_dim, dtype = 'int') * 1000000
    for i in range(len(g)):
        miller_to_g[miller[i, 0] + reciprocal_max_dim[0], miller[i, 1] + reciprocal_max_dim[1], miller[i, 2] + reciprocal_max_dim[2]] = i

    #increment the size of the real space grid until it is FFT-ready (only contains factors of 2, 3, or 5)
    for i in range( len(real_space_grid_dim) ):
        while np.any(np.union1d( factor_integer(real_space_grid_dim[i]), [2, 3, 5]) != [2, 3, 5]):
            real_space_grid_dim[i] += 1

    return g, g2, miller, reciprocal_max_dim, real_space_grid_dim, miller_to_g

# plane wave basis information
class plane_wave_basis():

    def __init__(self, energy_cutoff, a, h, k, cell):

        """

        plane wave basis information

        :param energy_cuttoff: kinetic energy cutoff (in atomic units)
        :param a: lattice vectors
        :param h: reciprocal lattice vectors
        :param k: k-points
        :param cell: the unit cell

        members:

        g: plane waves
        g2: square modulus of plane waves
        miller: the miller labels for plane waves, 
        reciprocal_max_dim: the maximum dimensions of the reciprocal basis,
        real_space_grid_dim: the number of real-space grid points, and 
        miller_to_g: a map between miller indices and a single index identifying the basis function
        n_plane_waves_per_k: number of plane wave basis functions per k-point
        kg_to_g: a map between basis functions for a given k-point and the original set of plane wave basis functions
        SI: structure factor

        """

        g, g2, miller, reciprocal_max_dim, real_space_grid_dim, miller_to_g = get_plane_wave_basis(energy_cutoff, a, h)
        n_plane_waves_per_k, kg_to_g = get_plane_waves_per_k(energy_cutoff, k, g)
        SI = get_SI(cell,g)

        self.g = g
        self.g2 = g2
        self.miller = miller
        self.reciprocal_max_dim = reciprocal_max_dim
        self.real_space_grid_dim = real_space_grid_dim
        self.miller_to_g = miller_to_g
        self.n_plane_waves_per_k = n_plane_waves_per_k
        self.kg_to_g = kg_to_g
        self.SI = SI

def get_plane_waves_per_k(energy_cutoff, k, g):

    """
   
    get dimension of plane wave basis for each k-point and a map between these
    the plane wave basis functions for a given k-point and the original list of plane waves

    :param energy_cutoff: the kinetic energy cutoff (in atomic units)
    :param k: the list of k-points
    :param g: the list plane wave basis functions
    :return n_plane_waves_per_k: the number of plane wave basis functions for each k-point
    :return n_plane_waves_per_k: the number of plane wave basis functions for each k-point

    """

    # n_plane_waves_per_k has dimension len(k) and indicates the number of orbital G vectors for each k-point

    n_plane_waves_per_k = np.zeros(len(k), dtype = 'int')

    for i in range(len(k)):
        for j in range(len(g)):
            # a basis function
            kgtmp = k[i] + g[j]

            # that basis function squared
            kg2tmp = np.dot(kgtmp,kgtmp)

            if(kg2tmp/2.0 <= energy_cutoff):
                n_plane_waves_per_k[i] += 1

    # kg_to_g maps basis for specific k-point to original plane wave basis
    kg_to_g = np.ones((len(k), np.amax(n_plane_waves_per_k)), dtype = 'int') * 1000000

    for i in range(len(k)):
        ind = 0
        for j in range(len(g)):

            # a basis function
            kgtmp = k[i]+g[j]

            # that basis function squared
            kg2tmp=np.dot(kgtmp,kgtmp)

            if(kg2tmp / 2.0 <= energy_cutoff):
                kg_to_g[i, ind] = j
                ind += 1

    return n_plane_waves_per_k, kg_to_g


def get_miller_indices(idx, basis):

    """

    get miller indices from composite label

    :param idx: composite index
    :param basis: plane wave basis information

    :return m1: miller index 1
    :return m2: miller index 2
    :return m3: miller index 3

    """

    m1 = basis.miller[idx,0]
    m2 = basis.miller[idx,1]
    m3 = basis.miller[idx,2]

    if m1 < 0:
        m1 = m1 + basis.real_space_grid_dim[0]
    if m2 < 0:
        m2 = m2 + basis.real_space_grid_dim[1]
    if m3 < 0:
        m3 = m3 + basis.real_space_grid_dim[2]

    return m1, m2, m3

def main():

    # material definition
    atom_type = 'Si'
    unit_type = 'diamond'
    lattice_constant = 10.26

    # kinetic energy cutoff
    energy_cutoff = 100.0 / 27.21138602

    # desired number of k-points in each direction
    n_k_points = [1, 1, 1]

    # get k-points, lattice vectors, reciprocal lattice vectors, cell volume, and cell itself
    k, a, b, omega, cell = get_cell_info(n_k_points, atom_type, unit_type, lattice_constant, energy_cutoff)

    # get plane wave basis information
    basis = plane_wave_basis(energy_cutoff, a, b, k, cell)

    # pseudopotential parameters (default Si)
    gth_params = gth_pseudopotential_parameters()

    # potential in reciprocal space
    vg = np.zeros(len(basis.g), dtype = 'float64')

    # maximum number of iterations
    maxiter = 100

    tiny = 1e-8

    rhog = np.zeros(len(basis.g), dtype = 'float64')

    scf_iter = 0

    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

    # electron-nucleus potential
    # TODO: make flexible for pseudopotential or not
    #valence_charges = np.array([gth_params.Zion, gth_params.Zion])
    valence_charges = cell.atom_charges()
    print(valence_charges)
    #exit()
    vne = get_nuclear_electronic_potential(cell, basis, omega, valence_charges = valence_charges)

    # number of bands, 
    total_charge = 0
    for I in range ( len(valence_charges) ):
        total_charge += valence_charges[I]
    nbands = int(total_charge / 2)

    print('total_charge',total_charge)
    print('nbands',nbands)
    

    # assuming nalpha = nbeta
    assert total_charge % 2 == 0

    print("")
    print("    ==> Begin SCF <==")
    print("")

    print("    %5s %20s" % ('iter','|drho|'))

    # begin SCF iterations
    for i in range(0,maxiter):

        new_rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
 
        # loop over k-points
        for j in range( len(k) ):

            # form fock matrix
            fock = np.zeros((basis.n_plane_waves_per_k[j], basis.n_plane_waves_per_k[j]), dtype = 'complex128')

            # get potential (dft + electron-electron)
            fock += get_potential(basis, k, j, vg)

            # get potential (nuclear-electronic)
            fock += get_potential(basis, k, j, vne)

            # get pseudopotential
            #fock += get_gth_pseudopotential(basis, gth_params, k, j, omega)

            # get kinetic energy
            kgtmp = k[j] + basis.g[basis.kg_to_g[j, :basis.n_plane_waves_per_k[j]]]
            diagonals = np.einsum('ij,ij->i', kgtmp, kgtmp) / 2.0 + fock.diagonal()
            np.fill_diagonal(fock, diagonals)

            #if scf_iter == 0 :
            #    assert np.isclose(24.132725572695584, np.linalg.norm(fock))

            # diagonalize fock matrix
            epsilon, C = scipy.linalg.eigh(fock, lower = False, eigvals=(0,2*nbands-1))
            print(epsilon)

            # build density 
            for pp in range(nbands):

                occ = np.zeros(basis.real_space_grid_dim,dtype = 'complex128')
                for tt in range( basis.n_plane_waves_per_k[j] ):
                    ik = basis.kg_to_g[j][tt]
                    occ[ get_miller_indices(ik, basis) ] = C[tt, pp]

                occ = ( 1.0 / np.sqrt(omega) ) * np.fft.fftn(occ)

                new_rho += ( 2.0 / len(k) ) * np.absolute(occ)**2.0

        #if scf_iter == 0 :
        #    assert np.isclose(2.539489449059902, np.linalg.norm(new_rho))

        # DFT potential
        xalpha = 2.0 / 3.0
        vr = -1.5 * xalpha * ( 3.0 * new_rho / np.pi )**( 1.0 / 3.0 )
        tmp = np.fft.ifftn(vr)
        for myg in range( len(basis.g) ):
            vg[myg]= np.real( tmp[ get_miller_indices(myg, basis) ] ) / omega
 
        #if scf_iter == 0:
        #    assert np.isclose(0.2847864818221838, np.linalg.norm(vg))

        # coulomb potential 
        tmp = np.fft.ifftn(new_rho)
        for myg in range( len(basis.g) ):
            rhog[myg] = np.real( tmp[ get_miller_indices(myg, basis) ] )
        vg += np.divide(4.0 * np.pi * rhog / omega, basis.g2, out=np.zeros_like(basis.g2), where = basis.g2 != 0.0) # save divide

        #if scf_iter == 0:
        #    assert np.isclose(0.32241081483652595, np.linalg.norm(vg))

        # convergence in density
        rho_diff = new_rho - rho
        rho_diff_norm = np.linalg.norm(rho_diff)

        #charge = ( omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.absolute(rho))
        #print(charge)

        print("    %5i %20.12lf %20.12lf %20.12lf %20.12lf %20.12lf" %  (scf_iter,rho_diff_norm,np.linalg.norm(rho),np.linalg.norm(new_rho),rho[1][1][1],rho[2][2][2]))
        rho = new_rho

        if ( rho_diff_norm < 1e-8 ) :
            break

        scf_iter += 1

    if scf_iter == maxiter:
        print('')
        print('    SCF did not converge.')
        print('')
    else:
        print('')
        print('    SCF converged!')
        print('')

    #print(done',np.linalg.norm(rhog),np.linalg.norm(vg))

if __name__ == "__main__":
    main()
