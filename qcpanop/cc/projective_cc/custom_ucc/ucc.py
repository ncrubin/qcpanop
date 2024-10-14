from typing import Optional

import numpy
import pyscf
from pyscf.cc.uccsd import UCCSD

from eris import CustomUERIs


def run_ucc(
    h1a: numpy.ndarray,
    h1b: numpy.ndarray,
    eri_aa: numpy.ndarray,
    eri_bb: numpy.ndarray,
    eri_ab: numpy.ndarray,
    nalpha: int,
    nbeta: int,
    #mf: pyscf.scf.hf,
    mol: pyscf.gto.Mole,
    with_triples: Optional[bool] = False,
):
    """
    Note, in order for this to work you need to set up a Mol object
    that has the minimum amount of information to determine the 
    number of alpha-electron and number of beta-electron. 

    For example, you need to set nelectron, spin and charge fields

    mol2_ = pyscf.M()
    mol2_.nelectron = A
    mol2_.spin = B
    mol2_.charge = C

    so that the nalpha, nbeta can be determined properly.
    """
    nmo = h1a.shape[0]
    mf = mol.UHF()
    mf.mo_coeff = (numpy.identity(nmo), numpy.identity(nmo))
    mf.mo_occ = numpy.asarray([
        [1]*nalpha + [0]*(nmo - nalpha),
        [1]*nbeta + [0]*(nmo - nbeta),
    ])
    mf.make_rdm1 = lambda *args: numpy.zeros((2, nmo, nmo))
    mf.get_veff = lambda *args: numpy.zeros((2, nmo, nmo))
    mf.energy_tot = lambda *args, **kwargs: 0
    ucc = UCCSD(mf)
    eris = CustomUERIs(h1a, h1b, eri_aa, eri_bb, eri_ab, nalpha, nbeta, mol)
    ucc.kernel(eris=eris)
    et = None
    if with_triples:
        et = ucc.ccsd_t(eris=eris)
    return ucc.e_corr, et
