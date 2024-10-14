import numpy
from pyscf.gto import Mole
from pyscf.cc.uccsd import _ChemistsERIs


class CustomUERIs(_ChemistsERIs):
    def __init__(self,
                 h1a: numpy.ndarray,
                 h1b: numpy.ndarray,
                 eri_aa: numpy.ndarray,
                 eri_bb: numpy.ndarray,
                 eri_ab: numpy.ndarray,
                 nalpha: int,
                 nbeta: int,
                 mol: Mole):
        nmo = h1a.shape[0]
        self.mol = mol
        if h1a.shape != (nmo, nmo):
            raise ValueError(f"Expected h1a shape of {(nmo, nmo)}, found {h1a.shape}")
        if h1b.shape != (nmo, nmo):
            raise ValueError(f"Expected h1b shape of {(nmo, nmo)}, found {h1b.shape}")
        if eri_aa.shape != (nmo, nmo, nmo, nmo):
            raise ValueError(f"Expected eri_aa shape of {(nmo, nmo, nmo, nmo)}, found {eri_aa.shape}")
        if eri_bb.shape != (nmo, nmo, nmo, nmo):
            raise ValueError(f"Expected eri_bb shape of {(nmo, nmo, nmo, nmo)}, found {eri_bb.shape}")
        if eri_ab.shape != (nmo, nmo, nmo, nmo):
            raise ValueError(f"Expected eri_ab shape of {(nmo, nmo, nmo, nmo)}, found {eri_ab.shape}")

        noa = nalpha
        nob = nbeta
        nva = nmo - nalpha
        nvb = nmo - nbeta
        self.oooo = eri_aa[:noa, :noa, :noa, :noa].copy()
        self.ovoo = eri_aa[:noa, noa:, :noa, :noa].copy()
        self.ovov = eri_aa[:noa, noa:, :noa, noa:].copy()
        self.oovv = eri_aa[:noa, :noa, noa:, noa:].copy()
        self.ovvo = eri_aa[:noa, noa:, noa:, :noa].copy()
        self.ovvv = eri_aa[:noa, noa:, noa:, noa:].copy()
        self.vvvv = eri_aa[noa:, noa:, noa:, noa:].copy()

        self.OOOO = eri_bb[:nob, :nob, :nob, :nob].copy()
        self.OVOO = eri_bb[:nob, nob:, :nob, :nob].copy()
        self.OVOV = eri_bb[:nob, nob:, :nob, nob:].copy()
        self.OOVV = eri_bb[:nob, :nob, nob:, nob:].copy()
        self.OVVO = eri_bb[:nob, nob:, nob:, :nob].copy()
        self.OVVV = eri_bb[:nob, nob:, nob:, nob:].copy()
        self.VVVV = eri_bb[nob:, nob:, nob:, nob:].copy()

        self.ooOO = eri_ab[:noa, :noa, :nob, :nob].copy()
        self.ovOO = eri_ab[:noa, noa:, :nob, :nob].copy()
        self.ovOV = eri_ab[:noa, noa:, :nob, nob:].copy()
        self.ooVV = eri_ab[:noa, :noa, nob:, nob:].copy()
        self.ovVO = eri_ab[:noa, noa:, nob:, :nob].copy()
        self.ovVV = eri_ab[:noa, noa:, nob:, nob:].copy()
        self.vvVV = eri_ab[noa:, noa:, nob:, nob:].copy()

        self.OVoo = eri_ab[:noa, :noa, :nob, nob:].copy().transpose((2, 3, 0, 1))
        self.OOvv = eri_ab[noa:, noa:, :nob, :nob].copy().transpose((2, 3, 0, 1))
        self.OVvo = eri_ab[noa:, :noa, :nob, nob:].copy().transpose((2, 3, 0, 1))
        self.OVvv = eri_ab[noa:, noa:, :nob, nob:].copy().transpose((2, 3, 0, 1))

        assert self.oooo.shape == (noa, noa, noa, noa)
        assert self.ovoo.shape == (noa, nva, noa, noa)
        assert self.ovov.shape == (noa, nva, noa, nva)
        assert self.oovv.shape == (noa, noa, nva, nva)
        assert self.ovvo.shape == (noa, nva, nva, noa)
        assert self.ovvv.shape == (noa, nva, nva, nva)
        assert self.vvvv.shape == (nva, nva, nva, nva)

        assert self.OOOO.shape == (nob, nob, nob, nob)
        assert self.OVOO.shape == (nob, nvb, nob, nob)
        assert self.OVOV.shape == (nob, nvb, nob, nvb)
        assert self.OOVV.shape == (nob, nob, nvb, nvb)
        assert self.OVVO.shape == (nob, nvb, nvb, nob)
        assert self.OVVV.shape == (nob, nvb, nvb, nvb)
        assert self.VVVV.shape == (nvb, nvb, nvb, nvb)

        assert self.ooOO.shape == (noa, noa, nob, nob)
        assert self.ovOO.shape == (noa, nva, nob, nob)
        assert self.ovOV.shape == (noa, nva, nob, nvb)
        assert self.ooVV.shape == (noa, noa, nvb, nvb)
        assert self.ovVO.shape == (noa, nva, nvb, nob)
        assert self.ovVV.shape == (noa, nva, nvb, nvb)
        assert self.vvVV.shape == (nva, nva, nvb, nvb)

        assert self.OVoo.shape == (nob, nvb, noa, noa)
        assert self.OOvv.shape == (nob, nob, nva, nva)
        assert self.OVvo.shape == (nob, nvb, nva, noa)
        assert self.OVvv.shape == (nob, nvb, nva, nva)

        Ja = numpy.einsum('pqii->pq', eri_aa[:, :, :noa, :noa]) \
            + numpy.einsum('pqii->pq', eri_ab[:, :, :nob, :nob])
        Jb = numpy.einsum('pqii->pq', eri_bb[:, :, :nob, :nob]) \
            + numpy.einsum('pqii->pq', eri_ab[:noa, :noa, :, :].transpose((2, 3, 0, 1)))
        Ka = numpy.einsum('piiq->pq', eri_aa[:, :noa, :noa, :])
        Kb = numpy.einsum('piiq->pq', eri_bb[:, :nob, :nob, :])

        self.focka = h1a + Ja - Ka
        self.fockb = h1b + Jb - Kb
        self.fock = (self.focka, self.fockb)
        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
