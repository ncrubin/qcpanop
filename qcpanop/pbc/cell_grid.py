"""
Use pyscf's cell infrastructure to define computational cell and grids

Implement Hamiltonian and SCF cycle
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



from pyscf.pbc import gto as pbcgto

def Get_Local_Pseudopotential_GTH(g2,c1,c2,c3,c4,rloc,Zion):

    """

    Construct and return local contribution to GTH pseudopotential

    :param g2: square modulus of plane wave basis functions
    :param c1: GTH pseudopotential parameter, c1
    :param c2: GTH pseudopotential parameter, c2
    :param c3: GTH pseudopotential parameter, c3
    :param c4: GTH pseudopotential parameter, c4
    :param rloc: GTH pseudopotential parameter, rloc
    :param Zion: ion charge
    :return: local contribution to GTH pseudopotential
    """

    vsg=np.zeros(len(g2),dtype='float64')
    largeind=g2>eps
    smallind=g2<=eps #|G|^2->0 limit
    g2=g2[largeind]
    rloc2=rloc*rloc
    rloc3=rloc*rloc2
    rloc4=rloc2*rloc2
    rloc6=rloc2*rloc4
    g4=g2*g2
    g6=g2*g4

    vsgl=np.exp(-g2*rloc2/2.)*(-4.*np.pi*Zion/g2+np.sqrt(8.*np.pi**3.)*rloc3*(c1+c2*(3.-g2*rloc2)+c3*(15.-10.*g2*rloc2+g4*rloc4)+c4*(105.-105.*g2*rloc2+21.*g4*rloc4-g6*rloc6)))
    vsgs=2.*np.pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*np.ip)*rloc+Zion) #|G|^2->0 limit
    vsg[largeind]=vsgl
    vsg[smallind]=vsgs

    return vsg/omega

def Get_Spherical_Harmonics_and_Projectors_GTH(gv,rl,lmax,imax):

    """

    Construct spherical harmonics and projectors for GTH pseudopotential

    :param gv: plane wave basis functions plus kpt
    :param rl: list of [rs, rp]
    :param lmax: maximum angular momentum, l
    :param imax: maximum i for projectors
    :return: spherical harmonics and projectors for GTH pseudopotential
    """

    rgv,thetagv,phigv=pbcgto.pseudo.pp.cart2polar(gv)

    mmax = 2*(lmax-1)+1
    gmax = len(gv)

    spherical_harmonics_lm = np.zeros((lmax,mmax,gmax),dtype='complex128')
    projector_li           = np.zeros((lmax,imax,gmax),dtype='complex128')

    for l in range(lmax):
        for m in range(-l,l+1):
            spherical_harmonics_lm[l,m+l,:] = scipy.special.sph_harm(m,l,phigv,thetagv)
        for i in range(imax):
            projector_li[l,i,:] = pbcgto.pseudo.pp.projG_li(rgv,l,i,rl[l])

    return spherical_harmonics_lm, projector_li

def Get_NonLocal_PseudoPotential_GTH(sphg,pg,gind,hgth):
    """

    Construct and return non-local contribution to GTH pseudopotential

    :param sphg: angular part of plane wave basis
    :param pg: projectors 
    :param gind: plane wave basis function label 
    :param hgth: GTH pseudopotential parameter, hlij
    :return: non-local contribution to GTH pseudopotential
    """

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

    return vsg/omega


def get_SI(cell, Gv=None):
    '''Calculate the structure factor (0D, 1D, 2D, 3D) for all atoms;
    see MH (3.34).

    S_{I}(G) = exp(iG.R_{I})

    Args:
        cell : instance of :class:`Cell`

        Gv : (N,3) array
            G vectors

    Returns:
        SI : (natm, ngrids) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.
    '''
    coords = cell.atom_coords()
    ngrids = np.prod(cell.mesh)
    if Gv is None or Gv.shape[0] == ngrids:
        # gets integer grid for Gv
        basex, basey, basez = cell.get_Gv_weights(cell.mesh)[1]
        b = cell.reciprocal_vectors()
        rb = np.dot(coords, b.T)
        SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
        SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
        SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
        SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
        SI = SI.reshape(-1,ngrids)
    else:
        SI = np.exp(-1j*np.dot(coords, Gv.T))
    return SI


def get_Gv(cell, mesh=None, **kwargs):
    '''Calculate three-dimensional G-vectors for the cell; see MH (3.8).

    Indices along each direction go as [0...N-1, -N...-1] to follow FFT convention.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        Gv : (ngrids, 3) ndarray of floats
            The array of G-vectors.
    '''
    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]

    gx = np.fft.fftfreq(mesh[0], 1./mesh[0])
    gy = np.fft.fftfreq(mesh[1], 1./mesh[1])
    gz = np.fft.fftfreq(mesh[2], 1./mesh[2])
    gxyz = lib.cartesian_prod((gx, gy, gz))

    b = cell.reciprocal_vectors()
    Gv = lib.ddot(gxyz, b)
    return Gv


def get_uniform_grids(cell, mesh=None, **kwargs):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    R = h N q

    h is the matrix of lattice vectors (bohr)
    N is diagonal with entry as 1/N_{x, y, z}
    q is a vector of ints [0, N_{x, y, z} - 1]
    N_{x, y, z} chosen s.t. N_{x, y, z} >= 2 * max(g_{x, y, z}) + 1
    where g is the tuple of

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.

    '''
    if mesh is None: mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    mesh = np.asarray(mesh, dtype=np.double)
    qv = lib.cartesian_prod([np.arange(x) for x in mesh])
    # 1/mesh is delta(XYZ) spacing
    # distributed over cell.lattice_vectors()
    # s.t. a_frac * mesh[0] = cell.lattice_vectors()
    a_frac = np.einsum('i,ij->ij', 1./mesh, cell.lattice_vectors())
    # now produce the coordinates in the cell
    coords = np.dot(qv, a_frac)
    return coords


def ke_matrix(cell, kpoint=np.array([0, 0, 0])):
    """
    construct kinetic-energy matrix at a particular k-point

    -0.5 nabla^{2} phi_{r}(g) = -0.5 (iG)^2 (1/sqrt(omega))exp(iG.r)
    =0.5G^2 phi_{r}(g)
    """
    diag_components = 0.5 * np.linalg.norm(cell.Gv + kpoint, axis=1)**2
    return np.diag(diag_components)


def potential_matrix(cell, kpoint=np.array([0, 0, 0])):
    """
    Calculate the potential energy matrix in planewaves
    :param cell:
    :param kpoint:
    :return:
    """
    pass


def get_nuc(mydf, kpts=None):
    """
    v(r) = \sum_{G}v(G)exp(iG.r)

    :param mydf:
    :param kpts:
    :return:
    """
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    cell = mydf.cell
    mesh = mydf.mesh  # mesh dimensions [2Ns + 1]
    charge = -cell.atom_charges()  # nuclear charge of atoms in cell
    Gv = cell.get_Gv(mesh)
    SI = get_SI(cell, Gv)
    assert SI.shape[1] == Gv.shape[0]

    rhoG = np.dot(charge, SI)  # this is Z_{I} * S_{I}

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)  # V(0) = 0 can be treated separately/
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    test_coulG = np.divide(4 * np.pi, absG2, out=np.zeros_like(absG2), where=absG2 != 0) # save divide
    assert np.allclose(coulG, test_coulG)

    vneG = rhoG * coulG
    # vneG / cell.vol  # Martin 12.16

    # this is for evaluating the potential on a real-space grid
    # potential = np.zeros((Gv.shape[0], Gv.shape[0]))
    # gx = np.fft.fftfreq(mesh[0], 1./mesh[0])
    # gy = np.fft.fftfreq(mesh[1], 1./mesh[1])
    # gz = np.fft.fftfreq(mesh[2], 1./mesh[2])
    # gxyz = lib.cartesian_prod((gx, gy, gz)).astype(int)
    # gxyz_dict = dict(zip([tuple(xx) for xx in gxyz], range(len(gxyz))))

    # for ridx, cidx in product(range(len(gxyz)), repeat=2):
    #     qprime_minus_q = tuple(gxyz[ridx] - gxyz[cidx])
    #     if qprime_minus_q in gxyz_dict.keys():
    #         potential[ridx, cidx] = vneG[gxyz_dict[qprime_minus_q]] / cell.vol
    # return potential


def get_pp(cell):
    from pyscf.pbc.gto.pseudo.pp import get_vlocG, get_gth_projG
    vlocG = get_vlocG(cell, Gv=cell.Gv)  # get VlocG for each atom! n_atoms x n_G

    hs, projs = get_gth_projG(cell, cell.Gv)
    #print(hs)
    #print(projs[0][0][0][0]) # [atm][l][m][i][ ngrids ]
    #print(len(projs))  # atoms
    #print(len(projs[0])) # num_non-local projectors angular moment l
    #print(projs[0][1])

    print(hs)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue
        print("sym ", symb)
        pp = cell._pseudo[symb]
        print(pp)
        for l, proj in enumerate(pp[5:]):
            print("l ", l, " proj ", proj)
            rl, nl, hl = proj
            hl = np.asarray(hl)
            h_mat_to_print = np.zeros((3, 3), dtype=hl.dtype)
            h_mat_to_print[:hl.shape[0], :hl.shape[1]] = hl
            print(h_mat_to_print)
                # pYlm = np.empty((nl,l*2+1,ngrids))
                # for k in range(nl):
                #     qkl = _qli(G_rad*rl, l, k)

    #exit()



def main():
    # cubic BCC structure (compare to ASE if curious)
    # cell = gto.M(a=np.eye(3) * 10,
    #             atom='C 0 0 1.1; H 0 0 0',
    #             basis='sto-3g',
    #             unit='angstrom',
    #             ke_cutoff=0.5)
    cell = gto.M(a=np.eye(3) * 10,
                 atom='Co 0 0 0',
                 basis='sto-3g',
                 pseudo='gth-blyp',
                 unit='angstrom',
                 ke_cutoff=0.5)
    cell.build()
    print("KE Cutoff from basis decay ",
          estimate_ke_cutoff(cell, cell.precision))
    print("KE cutoff from user ", cell.ke_cutoff)
    print("Mesh size form user KE-cutoff",
          cell.mesh)

    khf = scf.RHF(cell)
    fftdf = khf.with_df
    print(cell.a / BOHR)
    print("Cell geometry")
    print(cell.lattice_vectors())
    # Omega = Cell volumes
    print("Volume A^{3}", np.linalg.det(cell.a))
    print("Volume Bohr^{3}", np.linalg.det(cell.a / BOHR))
    assert np.isclose(np.linalg.det(cell.a / BOHR), cell.vol)

    # print("cell reciprocal lattice vecs")
    # print(cell.reciprocal_vectors())
    # print(2 * np.pi * np.linalg.inv((cell.a / BOHR).T))  # MH 3.6
    # this should be identity a_{i} . b_{j} = 2pi \delta_{i,j}
    assert np.allclose((cell.a / BOHR).dot(cell.reciprocal_vectors()) / (2 * np.pi), np.eye(3))

    # pick random R tuple and G tuple
    n1, n2, n3 = [np.random.randint(100) for _ in range(3)]
    m1, m2, m3 = [np.random.randint(100) for _ in range(3)]

    r_cvecs = cell.a / BOHR
    g_cvecs = cell.reciprocal_vectors()

    Rn = (r_cvecs.T @ np.diag([n1, n2, n3])).T
    Gm = (g_cvecs.T @ np.diag([m1, m2, m3])).T
    # should be all close to zero or 2 pi
    # print((Gm @ Rn) % (2 * np.pi))
    assert np.allclose(np.exp(1j * Gm @ Rn), np.ones((3, 3)))

    # print(get_Gv(cell=cell))
    # exit()

    # plane waves are
    # f(G) = (1/ sqrt(omega)) * exp(i G . r)
    # where G = 2pi(h.T)^{-1} g
    # where g = [i,j,k] of integers
    print(len(cell.get_Gv()))

    # so now we want to test this by making a periodic function in 3D
    # we will build up to this by starting with a periodic function in 1D
    # and then taking the Fourier transform

    # consider 1D with lattice vectors of 2 Angstrom (converted to bohr)
    L = cell.a[0, 0] / BOHR
    b = 2 * np.pi / L
    assert np.isclose(np.exp(1j * L * b), 1)
    # now all planewaves look like G = 2pi(h.T)^{-1} g where g is an integer
    # Energy units are hbar * nu = e^2 / (2 a0) where a0 is bohr radius
    # e^2 is 1 in au so Rydberg is  (k^2)/2
    num_pw = cell.mesh[0]
    gvals = np.arange(-num_pw // 2 + 1, num_pw // 2 + 1)  # symmetric around zero
    gx = np.fft.fftfreq(cell.mesh[0], 1./cell.mesh[0])
    print("zero centered int- grid")
    print(gvals)
    print("zero start int- grid")
    print(gx)
    Gv = b * gvals
    ke_Gv = 0.5 * np.abs(Gv)**2
    print("pw ke in 1D")
    print(ke_Gv)

    print("Real space mesh")
    rsmesh = get_uniform_grids(cell)
    print(len(rsmesh))

    # print(cell.Gv[28])
    # print(cell.Gv[28][0]**2 + cell.Gv[28][1]**2 + cell.Gv[28][2]**2)
    # print(0.5 * np.linalg.norm(cell.Gv[28])**2)
    # print(0.5 * np.linalg.norm(cell.Gv[28]) ** 2)

    # # print((cell.Gv + np.array([0, 1.2, 1.3])))
    # v_ion = get_nuc(fftdf)
    # ke = ke_matrix(cell)
    # exit()

    get_pp(cell)


    # get GTH pseudopotential matrix

    # define GTH parameters:

    # parameters for local contribution to pseudopotential (for Si)
    c1   = -7.336103
    c2   =  0.0
    c3   =  0.0
    c4   =  0.0
    rloc =  0.44
    Zion =  4.0

    # parameters for non-local contribution to pseudopotential (for Si)

    # hlij
    hgth = np.zeros((2,3,3),dtype='float64')
    hgth[0,0,0] = 5.906928
    hgth[0,1,1] = 3.258196
    hgth[0,0,1] = -0.5*np.sqrt(3./5.) * hgth[0,1,1]
    hgth[0,1,0] = -0.5*np.sqrt(3./5.) * hgth[0,1,1]
    hgth[1,0,0] = 2.727013

    # rs, rp, max l, max i
    r0   = 0.422738
    r1   = 0.484278
    rl   = [r0,r1]
    lmax = 2
    imax = 2

    # lattice constant
    lc = 10.26 

    sg = np.zeros(len(g),dtype='complex128')
    for i in range(len(g)):
        sg[i] = 2.0 * np.cos(np.dot(g[i],np.array([lc,lc,lc])/8.0))

    """ 
    need to define:

    k     = k-points [from GetMP() in gkclab/tickle]
    npw   = number of plane wave basis functions for each  k-point [from GetK() in gkclab/tickle]
    indgk = for each k-point, the index of the G vectors in contained in g [from GetK()]
    g     = the plane waves [from  GetGrids() in gkclab/tickle]
    g2    = the square modulus of the plane waves [from  GetGrids()]
    mill  = list of miller indices for G vectors [from GetGrids()]
    nm    = maximum dimension for reciprocal basis

    """

    for j in range(len(k)):
    
        gth_pseudopotential = np.zeros((npw[j],npw[j]),dtype='complex128')

        gkind = indgk[j,:npw[j]]
        gk    = g[gkind]

        sphg, pg = Get_Spherical_Harmonics_and_Projectors_GTH(k[j]+gk,rl,lmax,imax)

        for aa in range(npw[j]):

            ik    = indgk[j][aa]
            gdiff = mill[ik]-mill[gkind]+np.array(nm)
            inds  = indg[gdiff.T.tolist()]

            vsg_local = Get_Local_Pseudopotential_GTH(g2[inds],c1,c2,c3,c4,rloc,Zion)

            vsg_non_local = Get_NonLocal_PseudoPotential_GTH(sphg,pg,aa,hgth)

            gth_pseudopotential[aa,:] = ( vsg_local + vsg_non_local ) * sg[inds]

if __name__ == "__main__":
    main()
