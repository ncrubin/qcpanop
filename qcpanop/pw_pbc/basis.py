"""

plane wave basis information

"""

import numpy as np


from pyscf.lib.numpy_helper import cartesian_prod

from qcpanop.pw_pbc.pseudopotential import get_gth_pseudopotential_parameters

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

    # build all possible sets of miller indices
    ivals = np.arange(-reciprocal_max_dim_1, reciprocal_max_dim_1+1)
    jvals = np.arange(-reciprocal_max_dim_2, reciprocal_max_dim_2+1)
    kvals = np.arange(-reciprocal_max_dim_3, reciprocal_max_dim_3+1)
    ijk_vals = cartesian_prod([ivals, jvals, kvals])
    # get g-values i * b[0] + j * b[1] + k * b[2]
    gvals = ijk_vals @ b
    # compute g^2
    g2vals = np.einsum('ix,ix->i', gvals, gvals)
    # get items that are below cutoff
    density_cutoff_mask = np.where(g2vals/2 <= 4 * ke_cutoff)[0]
    gvals_below_cutoff = gvals[density_cutoff_mask]
    g2vals_below_cutoff = g2vals[density_cutoff_mask]
    miller_below_cutoff = ijk_vals[density_cutoff_mask]

    g = gvals_below_cutoff
    g2 = g2vals_below_cutoff
    miller = miller_below_cutoff

    # reciprocal_max_dim contains the maximum dimension for reciprocal basis
    reciprocal_max_dim = [int(np.amax(miller[:, 0])), int(np.amax(miller[:, 1])), int(np.amax(miller[:, 2]))]

    # real_space_grid_dim is the real space grid dimensions
    real_space_grid_dim = [2 * reciprocal_max_dim[0] + 1, 2 * reciprocal_max_dim[1] + 1, 2 * reciprocal_max_dim[2] + 1]

    # miller_to_g maps the miller indices to the index of G
    miller_to_g = np.ones(real_space_grid_dim, dtype = 'int')
    for i in range(len(g)):
        miller_to_g[miller[i, 0] + reciprocal_max_dim[0], miller[i, 1] + reciprocal_max_dim[1], miller[i, 2] + reciprocal_max_dim[2]] = i

    #increment the size of the real space grid until it is FFT-ready (only contains factors of 2, 3, or 5)
    for i in range( len(real_space_grid_dim) ):
        union_factors = np.union1d( factor_integer(real_space_grid_dim[i]), [2, 3, 5])
        while len(union_factors) > 3:
            real_space_grid_dim[i] += 1
            union_factors = np.union1d( factor_integer(real_space_grid_dim[i]), [2, 3, 5])

    return g, g2, miller, reciprocal_max_dim, real_space_grid_dim, miller_to_g

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
    kg_to_g = np.ones((len(k), np.amax(n_plane_waves_per_k)), dtype = 'int')

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

if __name__ == "__main__":
    import numpy
    from pyscf import gto
    from pyscf.pbc import gto as pgto
    cell = pgto.M(
        atom = '''C     0.      0.      0.    
                  C     0.8917  0.8917  0.8917
                  C     1.7834  1.7834  0.    
                  C     2.6751  2.6751  0.8917
                  C     1.7834  0.      1.7834
                  C     2.6751  0.8917  2.6751
                  C     0.      1.7834  1.7834
                  C     0.8917  2.6751  2.6751''',
        basis = {'C': gto.parse('''
    # Parse NWChem format basis string (see https://bse.pnl.gov/bse/portal).
    # Comment lines are ignored
    #BASIS SET: (6s,3p) -> [2s,1p]
    O    S
        130.7093200              0.15432897       
         23.8088610              0.53532814       
          6.4436083              0.44463454       
    O    SP
          5.0331513             -0.09996723             0.15591627       
          1.1695961              0.39951283             0.60768372       
          0.3803890              0.70011547             0.39195739       
                                    ''')},
        pseudo = 'gth-pade',
        a = numpy.eye(3)*3.5668,
        ke_cutoff=1000 / 27)
    
    cell.build()
    basis = plane_wave_basis(cell, 
                             ke_cutoff = cell.ke_cutoff,
                             n_kpts = [1, 1, 1],
                             nl_pp_use_legendre = True)
    print(basis.real_space_grid_dim)
