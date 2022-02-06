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

    exit()



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




if __name__ == "__main__":
    main()
