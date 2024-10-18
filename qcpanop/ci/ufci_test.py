from functools import reduce

import numpy
import pyscf
from pyscf import ao2mo

from .ufci import run_fci


def test_ufci():
    mol = pyscf.M(
        atom='He 0 0 0',
        basis='ccpvdz', charge=-1, spin=1)
    nalpha, nbeta = mol.nelec
    mf = mol.UHF()
    mf.kernel()
    cisolver = pyscf.fci.FCI(mf)
    ref = cisolver.kernel()[0]

    moa = mf.mo_coeff[0]
    mob = mf.mo_coeff[1]
    ha = reduce(numpy.dot, (moa.T, mf.get_hcore(), moa))
    hb = reduce(numpy.dot, (mob.T, mf.get_hcore(), mob))
    nmo = moa.shape[1]
    assert mob.shape[1] == nmo
    chem_eriaa = ao2mo.general(
        mol, (moa, moa, moa, moa), compact=False).reshape(nmo, nmo, nmo, nmo)
    chem_eribb = ao2mo.general(
        mol, (mob, mob, mob, mob), compact=False).reshape(nmo, nmo, nmo, nmo)
    chem_eriab = ao2mo.general(
        mol, (moa, moa, mob, mob), compact=False).reshape(nmo, nmo, nmo, nmo)

    out = run_fci(ha, hb, chem_eriaa, chem_eribb, chem_eriab, nalpha, nbeta)
    assert abs(out - ref) < 1e-8
