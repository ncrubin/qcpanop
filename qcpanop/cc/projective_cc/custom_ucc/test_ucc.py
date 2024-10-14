from functools import reduce
import numpy
import pyscf
from pyscf import ao2mo
from pyscf.cc import CCSD, UCCSD
from ucc import run_ucc


def test_uccsd_vs_rccsd():
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='ccpvdz')
    nalpha, nbeta = mol.nelec
    mf = mol.RHF()
    mf.kernel()
    cc = CCSD(mf)
    ref = cc.kernel()[0]
    ref_t = cc.ccsd_t()

    hcore = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))

    orb = mf.mo_coeff
    nmo = orb.shape[1]
    chem_eri = ao2mo.general(
        mol, (orb, orb, orb, orb), compact=False).reshape(nmo, nmo, nmo, nmo)

    out, out_t = run_ucc(hcore, hcore, chem_eri, chem_eri, chem_eri, nalpha, nbeta, mol, True)
    assert abs(ref - out) < 1e-6
    assert abs(ref_t - out_t) < 1e-6


def test_uccsd_vs_uccsd():
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='ccpvdz', charge=1, spin=1)
    nalpha, nbeta = mol.nelec
    mf = mol.UHF()
    mf.kernel()
    cc = UCCSD(mf)
    ref = cc.kernel()[0]
    ref_t = cc.ccsd_t()

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

    out, out_t = run_ucc(ha, hb, chem_eriaa, chem_eribb, chem_eriab, nalpha, nbeta, mol, True)
    assert abs(ref - out) < 1e-6
    assert abs(ref_t - out_t) < 1e-6


def test_fake_mol():
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='ccpvdz', charge=1, spin=1)

    mol2 = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='sto-3g', charge=1, spin=1)

    nalpha, nbeta = mol.nelec
    mf = mol.UHF()
    mf.kernel()
    cc = UCCSD(mf)
    ref = cc.kernel()[0]
    ref_t = cc.ccsd_t()

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

    out, out_t = run_ucc(ha, hb, chem_eriaa, chem_eribb, chem_eriab, nalpha, nbeta, mol2, True)
    assert abs(ref - out) < 1e-6
    assert abs(ref_t - out_t) < 1e-6

def test_fake_mol2():
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='ccpvdz', charge=1, spin=1)

    mol2 = pyscf.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='sto-3g', charge=1, spin=1)

    mol2_ = pyscf.M()
    mol2_.nelectron = mol2.nelectron
    mol2_.spin = mol2.spin
    mol2_.charge = mol2.charge

    nalpha, nbeta = mol.nelec
    mf = mol.UHF()
    mf.kernel()
    cc = UCCSD(mf)
    ref = cc.kernel()[0]
    ref_t = cc.ccsd_t()

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

    out, out_t = run_ucc(ha, hb, chem_eriaa, chem_eribb, chem_eriab, nalpha, nbeta, mol2_, True)
    assert abs(ref - out) < 1e-6
    assert abs(ref_t - out_t) < 1e-6


if __name__ == '__main__':
    # test_uccsd_vs_rccsd()
    # test_uccsd_vs_uccsd()
    # test_fake_mol()
    test_fake_mol2()
