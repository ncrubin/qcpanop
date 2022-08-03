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

from diis import DIIS

# GTH pseudopotential parameters
class gth_pseudopotential_parameters():

    def __init__(self, pp):

        """

        GTH pseudopotential parameter class

        :param pp: a pyscf pseudopotential object

        Attributes
        ----------

        Zion: list of the valence charges of all centers
        rloc: list of rloc pseudopotential parameters for all centers (local)
        local_cn: list of c coefficients (c1, c2, c3, c4) for all centers (local)
        lmax: list of maximum l value for all centers (nonlocal)
        imax: list of maximum i value for all centers (nonlocal)
        rl: list of rl values for all centers (nonlocal)
        hgth: list of h matrices for all centers (nonlocal)

        """

        # parameters for local contribution to pseudopotential

        # Zion
        self.Zion = 0.0
        for i in range(0,len(pp[0])):
            self.Zion += pp[0][i]

        # rloc
        self.rloc = pp[1]

        # c1, c2, c3, c4
        self.local_cn = np.zeros( (4), dtype='float64')
        for i in range(0,pp[2]):
            self.local_cn[i] = pp[3][i]
        
        # parameters for non-local contribution to pseudopotential

        # lmax
        self.lmax = pp[4]

        if self.lmax > 3:
            raise Exception('pseudopotentials currently only work with lmax <= 3. probably should change that.')

        # imax
        self.imax = 0
        for i in range(0,self.lmax):
            myi = pp[5+i][1]
            if myi > self.imax :
                self.imax = myi

        if self.imax > 3:
            raise Exception('pseudopotentials currently only work with imax <= 3. probably should change that.')

        # rl and h

        self.rl = np.zeros( (3), dtype='float64') # lmax = 3
        self.hgth = np.zeros((3, 3, 3),dtype='float64') # lmax = 3; imax = 3

        for l, proj in enumerate(pp[5:]):

            self.rl[l], nl, hl = proj
            hl = np.asarray(hl)
            my_h = np.zeros((3, 3), dtype=hl.dtype)
            if nl > 0:
                my_h[:hl.shape[0], :hl.shape[1]] = hl
                for i in range (0,3):
                    for j in range (0,3):
                        self.hgth[l, i, j] = my_h[i, j]

def get_gth_pseudopotential_parameters(cell):

    """

    get the GTH pseudopotential parameters

    :param cell: the unit cell

    :return gth_params: a list of pseudopotential parameters for each center

    """

    gth_params = []

    natom = len(cell._atom)

    for center in range (0,natom):

        element = cell._atom[center][0]

        pp = cell._pseudo[element]

        gth_params.append( gth_pseudopotential_parameters(pp) )

    return gth_params

 

def get_local_pseudopotential_gth(basis, tiny = 1e-8):


    """

    Construct and return local contribution to GTH pseudopotential

    :param basis: plane wave basis information

    :return: local contribution to GTH pseudopotential
    """

    vsg = np.zeros(len(basis.g2),dtype='complex128')

    for center in range (0, len(basis.gth_params) ) :

        c1 = basis.gth_params[center].local_cn[0]
        c2 = basis.gth_params[center].local_cn[1]
        c3 = basis.gth_params[center].local_cn[2]
        c4 = basis.gth_params[center].local_cn[3]
        rloc = basis.gth_params[center].rloc
        Zion = basis.gth_params[center].Zion

        largeind = basis.g2 > tiny
        smallind = basis.g2 <= tiny #|G|^2->0 limit

        my_g2 = basis.g2[largeind]
        SI_large = basis.SI[center,largeind]
        SI_small = basis.SI[center,smallind]

        rloc2 = rloc * rloc
        rloc3 = rloc * rloc2
        rloc4 = rloc2 * rloc2
        rloc6 = rloc2 * rloc4

        g4 = my_g2 * my_g2
        g6 = my_g2 * g4

        vsgl = SI_large * np.exp(-my_g2*rloc2/2.)*(-4.*np.pi*Zion/my_g2+np.sqrt(8.*np.pi**3.)*rloc3*(c1+c2*(3.-my_g2*rloc2)+c3*(15.-10.*my_g2*rloc2+g4*rloc4)+c4*(105.-105.*my_g2*rloc2+21.*g4*rloc4-g6*rloc6)))
        vsgs = SI_small * 2.*np.pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*np.pi)*rloc+Zion) #|G|^2->0 limit 

        vsg[largeind] += vsgl
        vsg[smallind] += vsgs

    return vsg / basis.omega

def get_spherical_harmonics_and_projectors_gth(gv, gth_params):

    """

    Construct spherical harmonics and projectors for GTH pseudopotential

    :param gv: plane wave basis functions plus kpt
    :param gth_params: GTH pseudopotential parameters
    :return spherical_harmonics: spherical harmonics in plane wave basis for k-point
    :return projectors_li: projectors for GTH pseudopotential in plane wave basis for k-point
    :return legendre: legendre polynomials 
    """

    # spherical polar representation of plane wave basis 
    rgv, thetagv, phigv = pbcgto.pseudo.pp.cart2polar(gv)
    
    gmax = len(gv)

    # number of atoms
    natom = len(gth_params)

    # maximum l and i hard-coded as 3. a pp with l,i exceeding this value
    # should be caught by the exceptions in gth_pseudopotential_parameters()
    lmax = 3 
    imax = 3 

    mmax = 2 * (lmax - 1) + 1

    # spherical harmonics should be independent of the center
    spherical_harmonics_lm = np.zeros((lmax, mmax, gmax),dtype='complex128')

    # legendre polynomials
    legendre = []

    # projectors should be center-dependent
    projector_li = np.zeros((natom, lmax, imax, gmax),dtype='complex128')


    for center in range (0,natom) :

        for l in range(lmax):
            for m in range(-l,l+1):
                spherical_harmonics_lm[l, m+l, :] = scipy.special.sph_harm(m, l, phigv, thetagv)
            for i in range(imax):
                projector_li[center, l, i, :] = pbcgto.pseudo.pp.projG_li(rgv, l, i, gth_params[center].rl[l])

            legendre.append( scipy.special.legendre(l) )

    return spherical_harmonics_lm, projector_li, legendre

def get_nonlocal_pseudopotential_gth(SI, sphg, pg, gind, gth_params, omega):

    """

    Construct and return non-local contribution to GTH pseudopotential

    :param SI: structure factor
    :param sphg: angular part of plane wave basis
    :param pg: projectors 
    :param gind: plane wave basis function label 
    :param gth_params: GTH pseudopotential parameters
    :param omega: unit cell volume
    :return: non-local contribution to GTH pseudopotential
    """

    # number of atoms
    natom = len(gth_params)

    vsg = 0.0

    for center in range (0, natom):

        my_h = gth_params[center].hgth
        my_pg = pg[center] 

        tmp_vsg = 0.0

        for l in range(0,gth_params[center].lmax):
            vsgij = vsgsp = 0.0
            for i in range(0,gth_params[center].imax):
                for j in range(0,gth_params[center].imax):
                    vsgij += my_pg[l, i, gind] * my_h[l, i, j] * my_pg[l,j,:]

            for m in range(-l,l+1):
                vsgsp += sphg[l,m+l,gind] * sphg[l,m+l,:].conj()

            tmp_vsg += vsgij * vsgsp

        # accumulate with structure factors
        vsg += tmp_vsg * SI[center][gind] * SI[center][:].conj()

    return vsg / omega

def get_nonlocal_pseudopotential_gth_legendre(SI, legendre, pg, gind, gth_params, omega, gv):

    """

    Construct and return non-local contribution to GTH pseudopotential

    :param SI: structure factor
    :param legendre: legendre polynomials
    :param pg: projectors 
    :param gind: plane wave basis function label 
    :param gth_params: GTH pseudopotential parameters
    :param omega: unit cell volume
    :param gv: plane wave basis functions plus kpt
    :return: non-local contribution to GTH pseudopotential
    """

    # number of atoms
    natom = len(gth_params)

    vsg = 0.0

    nrmgv = np.linalg.norm(gv, axis=1)

    for center in range (0, natom):

        my_h = gth_params[center].hgth
        my_pg = pg[center] 

        tmp_vsg = 0.0

        for l in range(0,gth_params[center].lmax):
            vsgij = vsgsp = 0.0
            for i in range(0,gth_params[center].imax):
                for j in range(0,gth_params[center].imax):
                    vsgij += my_pg[l, i, gind] * my_h[l, i, j] * my_pg[l,j,:]

            ratio = nrmgv[gind] * nrmgv[:]
            ratio = np.divide(gv[gind, 0] * gv[:, 0] + gv[gind, 1] * gv[:, 1] + gv[gind, 2] * gv[:, 2], ratio, out = np.zeros_like(ratio), where = ratio != 0.0)
            vsgsp = (2*l + 1)/(4 * np.pi) * np.polyval(legendre[l],  ratio[:, ])

            tmp_vsg += vsgij * vsgsp

        # accumulate with structure factors
        vsg += tmp_vsg * SI[center][gind] * SI[center][:].conj()

    return vsg / omega


def get_nonlocal_pseudopotential_matrix_elements(basis, kid, use_legendre = False):

    """

    get the GTH pseudopotential matrix for a given k-point

    :param basis: plane wave basis information
    :param kid: index for a given k-point
    :param use_legendre: flag to indicate use of spherical harmonics sum rule

    :return gth_pseudopotential: the GTH pseudopotential matrix in the plane wave basis for the orbitals, for this k-point (up to E <= ke_cutoff)

    """

    gth_pseudopotential = np.zeros((basis.n_plane_waves_per_k[kid], basis.n_plane_waves_per_k[kid]), dtype='complex128')

    # list of indices for basis functions for this k-point
    gkind = basis.kg_to_g[kid, :basis.n_plane_waves_per_k[kid]]

    # basis functions for this k-point
    gk = basis.g[gkind]

    # spherical harmonics and projectors in the basis of PWs for this K-point
    sphg, pg, legg = get_spherical_harmonics_and_projectors_gth(basis.kpts[kid] + gk, basis.gth_params)

    if use_legendre: 
        for aa in range(basis.n_plane_waves_per_k[kid]):
            # get a row of the pseudopotential matrix for this k-point
            gth_pseudopotential[aa, aa:] = get_nonlocal_pseudopotential_gth_legendre(basis.SI[:,gkind], legg, pg, aa, basis.gth_params, basis.omega, basis.kpts[kid] + gk)[aa:]
    else:
        for aa in range(basis.n_plane_waves_per_k[kid]):
            # get a row of the pseudopotential matrix for this k-point
            gth_pseudopotential[aa, aa:] = get_nonlocal_pseudopotential_gth(basis.SI[:,gkind], sphg, pg, aa, basis.gth_params, basis.omega)[aa:]

    return gth_pseudopotential

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

def get_SI(cell, Gv=None):

    """
    Calculate the structure factor for all atoms; see MH (3.34).

    Args:
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

    # structure factor
    SI = get_SI(cell, basis.g)

    assert SI.shape[1] == basis.g.shape[0]

    # this is Z_{I} * S_{I}
    rhoG = np.dot(charges, SI)  

    coulG = np.divide(4.0 * np.pi, basis.g2, out = np.zeros_like(basis.g2), where = basis.g2 != 0.0)

    vneG = - rhoG * coulG / basis.omega

    return vneG

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

def get_plane_wave_basis(ke_cutoff, a, b):

    """
    
    get plane wave basis functions and indices

    :param ke_cutoff: kinetic energy cutoff (in atomic units)
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
    reciprocal_max_dim_1 = int( np.ceil( (np.sqrt(2.0*4.0*ke_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[0]) + 1.0))
    reciprocal_max_dim_2 = int( np.ceil( (np.sqrt(2.0*4.0*ke_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[1]) + 1.0))
    reciprocal_max_dim_3 = int( np.ceil( (np.sqrt(2.0*4.0*ke_cutoff) / (2.0 * np.pi) ) * np.linalg.norm(a[2]) + 1.0))

    # g, g2, and miller indices
    g = np.empty(shape=[0,3],dtype='float64')
    g2 = np.empty(shape=[0,0],dtype='complex128')
    miller = np.empty(shape=[0,3],dtype='int')

    for i in np.arange(-reciprocal_max_dim_1, reciprocal_max_dim_1+1):
        for j in np.arange(-reciprocal_max_dim_2, reciprocal_max_dim_2+1):
            for k in np.arange(-reciprocal_max_dim_3, reciprocal_max_dim_3+1):

                # G vector
                gtmp = i * b[0] + j * b[1] + k * b[2]

                # |G|^2
                g2tmp = np.dot(gtmp, gtmp)

                # ke_cutoff for density is 4 times ke_cutoff for orbitals
                if (g2tmp/2.0 <= 4.0 * ke_cutoff):

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

    def __init__(self, cell, ke_cutoff = 18.374661240427326, n_kpts = [1, 1, 1], nl_pp_use_legendre = False):

        """

        plane wave basis information

        :param cell: the unit cell
        :param ke_cutoff: kinetic energy cutoff (in atomic units), default = 500 eV
        :param n_kpts: number of k-points
        :param nl_pp_use_legendre: use sum rule expression for spherical harmonics?

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
        use_pseudopotential: use a pseudopotential for all atoms?
        gth_params: GTH pseudopotential parameters
        omega: the unit cell volume
        kpts: the k-points
        ke_cutoff: kinetic energy cutoff (atomic units)
        n_kpts: number of k-points
        charge: charge

        """

        a = cell.a
        h = cell.reciprocal_vectors()

        self.nl_pp_use_legendre = nl_pp_use_legendre

        # get k-points
        self.kpts = cell.make_kpts(n_kpts, wrap_around = True)

        g, g2, miller, reciprocal_max_dim, real_space_grid_dim, miller_to_g = get_plane_wave_basis(ke_cutoff, a, h)
        n_plane_waves_per_k, kg_to_g = get_plane_waves_per_k(ke_cutoff, self.kpts, g)
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
        self.ke_cutoff = ke_cutoff
        self.n_kpts = n_kpts

        self.use_pseudopotential = False
        if cell.pseudo is not None :
            self.use_pseudopotential = True

        self.gth_params = None
        if ( self.use_pseudopotential ):
            self.gth_params = get_gth_pseudopotential_parameters(cell)

        self.omega = np.linalg.det(cell.a)

        self.charge = cell.charge


def get_plane_waves_per_k(ke_cutoff, k, g):

    """
   
    get dimension of plane wave basis for each k-point and a map between these
    the plane wave basis functions for a given k-point and the original list of plane waves

    :param ke_cutoff: the kinetic energy cutoff (in atomic units)
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

            if(kg2tmp/2.0 <= ke_cutoff):
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

            if(kg2tmp / 2.0 <= ke_cutoff):
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

def get_density(basis, C, N, kid):
    """

    get real-space density from molecular orbital coefficients

    :param basis: plane wave basis information
    :param C: molecular orbital coefficients
    :param N: the number of electrons
    :param kid: index for a given k-point

    :return rho: the density

    """

    rho = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
    for pp in range(N):

        occ = np.zeros(basis.real_space_grid_dim,dtype = 'complex128')

        for tt in range( basis.n_plane_waves_per_k[kid] ):

            ik = basis.kg_to_g[kid][tt]
            occ[ get_miller_indices(ik, basis) ] = C[tt, pp]

        occ = ( 1.0 / np.sqrt(basis.omega) ) * np.fft.fftn(occ)

        rho += np.absolute(occ)**2.0

    return ( 1.0 / len(basis.kpts) ) * rho

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


def pw_uks(cell, basis, xc = 'lda', guess_mix = True):

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

    if xc != 'lda':
        raise Exception("pw_uks only supports xc = 'lda' for now")

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

        new_rho_alpha = np.zeros(basis.real_space_grid_dim, dtype = 'float64')
        new_rho_beta = np.zeros(basis.real_space_grid_dim, dtype = 'float64')

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
            new_rho_alpha += get_density(basis, Calpha, nalpha, kid)

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
            new_rho_beta += get_density(basis, Cbeta, nbeta, kid)

        # LSDA XC energy
        cx = - 3.0 / 4.0 * ( 3.0 / np.pi )**( 1.0 / 3.0 ) 
        xc_energy = cx * 2.0 ** ( 1.0 / 3.0 ) * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.power(new_rho_alpha, 4.0/3.0))
        xc_energy += cx * 2.0 ** ( 1.0 / 3.0 ) * ( basis.omega / ( basis.real_space_grid_dim[0] * basis.real_space_grid_dim[1] * basis.real_space_grid_dim[2] ) ) * np.sum(np.power(new_rho_beta, 4.0/3.0))

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

        # LSDA potential
        vr_alpha = 4.0 / 3.0 * cx * 2.0 ** ( 1.0 / 3.0 ) * np.power( rho_alpha , 1.0 / 3.0 )
        vr_beta  = 4.0 / 3.0 * cx * 2.0 ** ( 1.0 / 3.0 ) * np.power( rho_beta , 1.0 / 3.0 )

        tmp = np.fft.ifftn(vr_alpha)
        for myg in range( len(basis.g) ):
            v_xc_alpha[myg] = tmp[ get_miller_indices(myg, basis) ]

        tmp = np.fft.ifftn(vr_beta)
        for myg in range( len(basis.g) ):
            v_xc_beta[myg] = tmp[ get_miller_indices(myg, basis) ]

        # total energy
        new_total_energy = np.real(one_electron_energy) + np.real(coulomb_energy) + xc_energy + enuc

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
    print('    xc energy:                %20.12lf' % ( xc_energy ) )
    print('')
    print('    total energy:             %20.12lf' % ( np.real(one_electron_energy) + np.real(coulomb_energy) + xc_energy + enuc ) )
    print('')

    assert(np.isclose( np.real(one_electron_energy) + np.real(coulomb_energy) + xc_energy + enuc, -9.802901383306) )


