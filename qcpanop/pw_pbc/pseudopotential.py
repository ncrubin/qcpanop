"""

GTH pseudopotentials

"""

import scipy
import numpy as np

from pyscf.pbc import gto as pbcgto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

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

        k2 = my_g2 * rloc2
        k4 = k2 * k2
        k6 = k2 * k4

        rloc4 = rloc2 * rloc2
        rloc6 = rloc2 * rloc4

        vsgl = SI_large * np.exp(-k2/2.0) * (-4.0 * np.pi * Zion / my_g2 + np.sqrt(8.0 * np.pi**3.0) * rloc3 * (c1 + c2 * (3.0 - k2) + c3 * (15.0 - 10.0 * k2 + k4) + c4 * (105.0 - 105.0 * k2 + 21.0 * k4 - k6) ) )

        #|G|^2->0 limit 
        vsgs = SI_small * 2.0 * np.pi * rloc2 * ( (c1 + 3.0 * (c2 + 5.0 * (c3 + 7.0 * c4) ) ) * np.sqrt(2.0 * np.pi) * rloc + Zion) 

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

    # rgv_test = np.sqrt(np.einsum('ik,ik->i', gv, gv))
    # assert np.allclose(rgv_test, rgv) # ||k|| which will go into the projectors

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

    rgv, thetagv, phigv = pbcgto.pseudo.pp.cart2polar(gv)

    for center in range (0, natom):

        my_h = gth_params[center].hgth
        my_pg = pg[center] 

        tmp_vsg = 0.0

        for l in range(0,gth_params[center].lmax):
            vsgij = vsgsp = 0.0
            for i in range(0,gth_params[center].imax):
                for j in range(0,gth_params[center].imax):
                    vsgij += my_pg[l, i, gind] * my_h[l, i, j] * my_pg[l,j,:]

            #ratio = nrmgv[gind] * nrmgv[:]
            #ratio = np.divide(gv[gind, 0] * gv[:, 0] + gv[gind, 1] * gv[:, 1] + gv[gind, 2] * gv[:, 2], ratio, out = np.zeros_like(ratio), where = ratio != 0.0)
            #vsgsp = (2*l + 1)/(4 * np.pi) * np.polyval(legendre[l],  ratio[:, ])

            cos_gamma = np.cos(thetagv[gind]) * np.cos(thetagv[:]) + np.sin(thetagv[gind]) * np.sin(thetagv[:]) * np.cos( phigv[gind] - phigv[:] )
            vsgsp = (2*l + 1)/(4 * np.pi) * np.polyval(legendre[l],  cos_gamma )

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


