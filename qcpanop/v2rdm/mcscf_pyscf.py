import os
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"

import psi4
import sys
# sys.path.insert(0, '/usr/local/google/home/nickrubin/dev/hilbert')
sys.path.insert(0, '/usr/local/google/home/nickrubin/dev')
import hilbert

import numpy
import pyscf

from itertools import product


class MP2AsFCISolver(object):
    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        # Kernel takes the set of integrals from the current set of orbitals
        fakemol = pyscf.M(verbose=0)
        fakemol.nelectron = sum(nelec)
        fake_hf = fakemol.RHF()
        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        # Build an SCF object fake_hf without SCF iterations to perform MP2
        fake_hf.mo_coeff = numpy.eye(norb)
        fake_hf.mo_occ = numpy.zeros(norb)
        fake_hf.mo_occ[:fakemol.nelectron//2] = 2
        self.mp2 = fake_hf.MP2().run()
        return self.mp2.e_tot + ecore, self.mp2.t2

    def make_rdm12(self, t2, norb, nelec):
        dm1 = self.mp2.make_rdm1(t2)
        dm2 = self.mp2.make_rdm2(t2)
        return dm1, dm2


class FCIAsFCISolver(object):
    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        # Kernel takes the set of integrals from the current set of orbitals
        fakemol = pyscf.M(verbose=0)
        fakemol.nelectron = sum(nelec)
        fake_hf = fakemol.RHF()
        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        # Build an SCF object fake_hf without SCF iterations to perform MP2
        fake_hf.mo_coeff = numpy.eye(norb)
        fake_hf.mo_occ = numpy.zeros(norb)
        fake_hf.mo_occ[:fakemol.nelectron//2] = 2
        myci = pyscf.fci.FCI(fake_hf)
        res = myci.run(ci0=ci0)
        self.ci = res
        return self.ci.e_tot + ecore, self.ci.ci

    def make_rdm12(self, civec, ncas, nelec):
        dm1 = self.ci.make_rdm1(civec, ncas, nelec)
        dm2 = self.ci.make_rdm2(civec, ncas, nelec)
        return dm1, dm2


class V2RDMAsFCISolver(object):
    def __init__(self, rconvergence=1.e-5, econvergence=1e-4, positivity='dqg', sdp_solver='bpsdp', maxiter=20_000):
        # hilbert options
        if sdp_solver.lower() == 'bpsdp':
            psi4.set_module_options('hilbert', {
                'positivity': positivity,
                'r_convergence': rconvergence,
                'e_convergence': econvergence,
                'maxiter': maxiter,
                'print': 5
            })
        elif sdp_solver.lower() == 'rrsdp':
            psi4.set_module_options('hilbert', {
                'sdp_solver':      'rrsdp',
                'positivity': positivity,
                'r_convergence': rconvergence,
                'e_convergence': econvergence,
                'maxiter': maxiter,
            })
        else:
            raise KeyError("Invalid value of sdp_solver: {}. Must enter bpsdp or rrsdp".format(sdp_solver))

        # grab options object
        options = psi4.core.get_options()
        options.set_current_module('HILBERT')
        self.options = options

        # this will be the returned variable
        self.rdms = (None, None)

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        # Kernel takes the set of integrals from the current set of orbitals
        fakemol = pyscf.M(verbose=0)
        fakemol.nelectron = sum(nelec)
        fake_hf = fakemol.RHF()
        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        # Build an SCF object fake_hf without SCF iterations to perform MP2
        fake_hf.mo_coeff = numpy.eye(norb)
        fake_hf.mo_occ = numpy.zeros(norb)
        fake_hf.mo_occ[:fakemol.nelectron//2] = 2

        # this is where we need to add our v2rdm code.
        nalpha, nbeta = nelec[0], nelec[1]
        nmo = norb

        v2rdm_pyscf = hilbert.v2RDMHelper(nalpha, nbeta, nmo, h1.flatten(), h2.flatten(), self.options)
        current_energy = v2rdm_pyscf.compute_energy()

        # tpdm = numpy.array(v2rdm_pyscf.get_tpdm())  # spin-summed tpdm. This function is broken
        opdm = numpy.array(v2rdm_pyscf.get_opdm())  # spin-summed opdm

        tpdm_vec = v2rdm_pyscf.get_tpdm_sparse('AA')
        tpdm_aa = numpy.zeros((nmo, nmo, nmo, nmo))
        for cnt in range(len(tpdm_vec)):
            i, j, k, l = tpdm_vec[cnt].i, tpdm_vec[cnt].j, tpdm_vec[cnt].k, tpdm_vec[cnt].l,
            tpdm_aa[i, j, k, l] = tpdm_vec[cnt].value

        tpdm_vec = v2rdm_pyscf.get_tpdm_sparse('BB')
        tpdm_bb = numpy.zeros((nmo, nmo, nmo, nmo))
        for cnt in range(len(tpdm_vec)):
            i, j, k, l = tpdm_vec[cnt].i, tpdm_vec[cnt].j, tpdm_vec[cnt].k, tpdm_vec[cnt].l,
            tpdm_bb[i, j, k, l] = tpdm_vec[cnt].value

        tpdm_vec = v2rdm_pyscf.get_tpdm_sparse('AB')
        tpdm_ab = numpy.zeros((nmo, nmo, nmo, nmo))
        for cnt in range(len(tpdm_vec)):
            i, j, k, l = tpdm_vec[cnt].i, tpdm_vec[cnt].j, tpdm_vec[cnt].k, tpdm_vec[cnt].l,
            tpdm_ab[i, j, k, l] = tpdm_vec[cnt].value

        tpdm_vec = v2rdm_pyscf.get_tpdm_sparse('BA')
        tpdm_ba = numpy.zeros((nmo, nmo, nmo, nmo))
        for cnt in range(len(tpdm_vec)):
            i, j, k, l = tpdm_vec[cnt].i, tpdm_vec[cnt].j, tpdm_vec[cnt].k, tpdm_vec[cnt].l,
            tpdm_ba[i, j, k, l] = tpdm_vec[cnt].value

        stpdm = tpdm_ab + tpdm_ba + tpdm_aa + tpdm_bb
        stpdm = stpdm.transpose(0, 2, 1, 3)  # this is the ordering required for pyscf
        # see here for pyscfs def ofps in-traced
        # https://pyscf.org/pyscf_api_docs/pyscf.fci.html?highlight=make_rdm2#pyscf.fci.selected_ci.make_rdm2

        self.tpdm_aa = tpdm_aa
        self.tpdm_ab = tpdm_ab
        self.tpdm_ba = tpdm_ba
        self.tpdm_bb = tpdm_bb
        self.spin_summed_tpdm = stpdm
        self.spin_summed_opdm = opdm
        self.rdms = (opdm, stpdm)

        return current_energy + ecore, self.rdms

    def make_rdm12(self, spin_summed_rdms, ncas, nelec):
        dm1 = spin_summed_rdms[0]
        dm2 = spin_summed_rdms[1]
        return dm1, dm2


if __name__ == "__main__":
    mol = pyscf.M(atom='N 0 0 0; N 0 0 1.1', basis='ccpvtz')
    mf = mol.RHF().run()
    print()

    print()
    print("Running v2rdm CASSCF")
    v2rdmfci = pyscf.mcscf.CASSCF(mf, 2, 2)
    v2rdmfci.fcisolver = V2RDMAsFCISolver(rconvergence=1.E-8, econvergence=1.E-8, maxiter=5)
    v2rdmfci.kernel()
    print("Finished running v2rdm casscf")
    print(v2rdmfci.e_tot)

    print("pyscf starting casscf")
    pyscffci = pyscf.mcscf.CASSCF(mf, 2, 2)
    pyscffci.kernel()
    print("pyscf fci")
    print(pyscffci.e_tot)

    assert numpy.isclose(pyscffci.e_tot, v2rdmfci.e_tot)