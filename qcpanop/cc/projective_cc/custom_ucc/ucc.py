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
    mf: pyscf.scf.hf,
    with_triples: Optional[bool] = False,
):

    ucc = UCCSD(mf)
    eris = CustomUERIs(h1a, h1b, eri_aa, eri_bb, eri_ab, nalpha, nbeta, mf.mol)
    ucc.kernel(eris=eris)
    et = None
    if with_triples:
        et = ucc.ccsd_t()
    return ucc.e_corr, et
