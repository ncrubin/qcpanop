"""
Density fitted-MP2 taken from pyscf and annotated by @nickrubin


"""
# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import einsum

import numpy
# from numpy import einsum
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
from pyscf.mp.mp2 import make_rdm1, make_rdm2
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)



def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen == 0 or mp.frozen is None)

    ####################################################################
    # this is extra safe re-initialization. eris ChemistERIs should hold
    # only the orbital energies and not the 4-tensor eri
    #####################################################################
    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    ##########################################
    #
    # This is initializing the _cderi
    # field.  Is his really what we want?
    #
    ##########################################
    # mp.with_df is a pyscf.df.df.DF object
    # calling this builds the integrals
    naux = mp.with_df.get_naoaux()
    # mo_energy[:nocc, None] forms a column vector of occupied energies
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    ############################################################
    #
    #   Initialize T2 vector which if we need forces we will
    #     need later. Watch the precision of this object
    #
    #########################################################3#
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
    else:
        t2 = None

    ##########################################
    #
    #     Load a tensor of Cholesky tensors
    #
    ##########################################
    Lov = numpy.empty((naux, nocc*nvir))
    p1 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        p0, p1 = p1, p1 + qov.shape[0]
        Lov[p0:p1] = qov
    emp2 = 0

    ###################################
    #
    #   loop over a single occupied
    #     index.
    #
    #####################################
    for i in range(nocc):
        buf = numpy.dot(Lov[:, i * nvir:(i + 1) * nvir].T,
                        Lov).reshape(nvir, nocc, nvir)
        gi = numpy.array(buf, copy=False)
        gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
        t2i = gi / lib.direct_sum('jb+a->jba', eia, eia[i])
        emp2 += einsum('jab,jab', t2i, gi) * 2
        emp2 -= einsum('jab,jba', t2i, gi)
        if with_t2:
            t2[i] = t2i

    return emp2, t2


class DFMP2(mp2.MP2):
    """
    Density-Fitted MP2 from pyscf.  Overwrites appropriate functions

    Sharp Edge: Does this work with non-canonical orbitals?
    Forces are not implemented?

    Actual work is done in kernel function which is called by init_amps from
    mp2.MP2 class for canonical orbitals.

    Functions overwritten:
        reset : reset now resets with_df attribute
        ao2mo : this one is important.  since we don't want to actually do
                the integral rotation we return the ChemistsERI object which
                sets the fock matrix (depending on if canonical or non-canonical
                orbitals) and HF energy.  Does not produce any eris!

    """
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')  # we might want to check this and then do the reorder
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]  # allocated memory in MB
        max_memory = max(2000, self.max_memory*.9-mem_now)  # on  my machine this is 3.6 GB self.max_memory is 4GB.  We will likely need to change this
        # blockdim sets blocksize to read from disk of the 3C-ints
        # is this needed?
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        # Initialize only the mo_coeff
        eris._common_init_(self, mo_coeff)
        return eris

    def make_rdm1(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm1(self, t2, ao_repr=ao_repr)

    def make_rdm2(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm2(self, t2, ao_repr=ao_repr)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = DFMP2

from pyscf import scf
scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)
scf.rohf.ROHF.DFMP2 = None
scf.uhf.UHF.DFMP2 = None

del(WITH_T2)

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from qcpanop.mol_io import read_xyz

    sym, coords = read_xyz(xyz_file="dcb_tweezer.xyz")
    geom_list = list(zip(sym, coords))

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    # mol.atom = geom_list

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)


