"""Test CCD code against pyscf's ccd hack"""
import openfermion as of
from qcpanop.cc.ccd import CCD


def test_ccd():
    from pyscf import gto, scf, cc
    from openfermionpyscf import run_pyscf

    geometry = [['H', (0., 0., 0.)], ['B', (0., 0., 1.6)]]
    mol = gto.M(
        atom=geometry,
        basis='cc-pvdz')

    mf = scf.RHF(mol).run()

    mycc = cc.CCSD(mf)
    mycc.frozen = 1
    old_update_amps = mycc.update_amps

    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return t1 * 0, t2

    mycc.update_amps = update_amps
    mycc.kernel()

    print('CCD correlation energy', mycc.e_corr)


    molecule = of.MolecularData(geometry=mol.atom,
                                basis=mol.basis,
                                charge=mol.charge,
                                multiplicity=mol.spin + 1)
    molecule = run_pyscf(molecule)
    cc = CCD(molecule=molecule)
    cc.solve_for_amplitudes()
    print('NCR-CCD energy ', cc.ccd_energy + molecule.nuclear_repulsion)
    print('NCR-CCD correlation energy ', cc.ccd_energy + molecule.nuclear_repulsion - cc.scf_energy)

    cc.pccd_solve()
    print('NCR-pCCD energy ', cc.ccd_energy + molecule.nuclear_repulsion)
    print('NCR-pCCD correlation energy ', cc.ccd_energy + molecule.nuclear_repulsion - cc.scf_energy)

    from qcpanop.cc.pccd import pCCD
    pccd = pCCD(molecule=molecule)
    pccd.compute_energy()


if __name__ == "__main__":
    test_ccd()