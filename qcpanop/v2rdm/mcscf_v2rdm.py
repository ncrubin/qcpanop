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


class V2RDMAsFCISolver(object):
    def __init__(self, rconvergence=1.e-5, econvergence=1e-4, positivity='dqg', sdp_solver='bpsdp', maxiter=20_000,
                 spin=None):
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

        if spin is None:
            self.spin = 0  # 2S not 2S + 1. This is how pyscf stores multiplicity
            self.sz = 0

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        # Kernel takes the set of integrals from the current set of orbitals
        fakemol = pyscf.M(verbose=0)
        fakemol.nelectron = sum(nelec)
        if nelec[0] == nelec[1]:
            fake_hf = fakemol.RHF()
        else:
            fake_hf = fakemol.ROHF()

        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        # Build an SCF object fake_hf without SCF iterations to perform MP2
        fake_hf.mo_coeff = numpy.eye(norb)
        fake_hf.mo_occ = numpy.zeros(norb)

        # build the correct alpha beta spins.
        nalpha, nbeta = nelec[0], nelec[1]
        nmo = norb
        alpha_diag = [1] * nalpha + [0] * (nmo - nalpha)
        beta_diag = [1] * nbeta + [0] * (nmo - nbeta)
        fake_hf.mo_occ = numpy.array(alpha_diag) + numpy.array(beta_diag)


        # run 2-RDM code. Remember that S = ms = 1/2(na - nb)--i.e. always high spin
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
    pyscffci.verbose = 5
    pyscffci.kernel()
    print("pyscf fci")
    print(pyscffci.e_tot)

    assert numpy.isclose(pyscffci.e_tot, v2rdmfci.e_tot)