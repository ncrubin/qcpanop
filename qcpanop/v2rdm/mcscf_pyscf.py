import numpy
import pyscf


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

if __name__ == "__main__":
    mol = pyscf.M(atom='N 0 0 0; N 0 0 1.1', basis='ccpvtz')
    mf = mol.RHF().run()
    mf.MP2().run()
    print()
    mcfci = pyscf.mcscf.CASSCF(mf, 12, 6)
    mcfci.fcisolver = FCIAsFCISolver()
    # mcfci.verbose = 2
    mcfci.kernel()
    print("Finished calculating")
    print()
    print("pyscf starting casscf")
    pyscffci = pyscf.mcscf.CASSCF(mf, 12, 6)
    pyscffci.kernel()
    print("pyscf fci")
    exit()
    exit()
    # Put in the active space all orbitals of the system
    mc = pyscf.mcscf.CASSCF(mf, mol.nao, mol.nelectron)
    mc.fcisolver = MP2AsFCISolver()
    # Internal rotation inside the active space needs to be enabled
    mc.internal_rotation = True
    mc.kernel()