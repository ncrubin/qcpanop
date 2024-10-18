import numpy
import pyscf


def run_fci(
    h1a: numpy.ndarray,
    h1b: numpy.ndarray,
    eri_aa: numpy.ndarray,
    eri_bb: numpy.ndarray,
    eri_ab: numpy.ndarray,
    nalpha: int,
    nbeta: int,
) -> float:
    nmo = h1a.shape[0]
    return pyscf.fci.direct_uhf.kernel(
        (h1a, h1b), (eri_aa, eri_ab, eri_bb), nmo, (nalpha, nbeta))[0]
