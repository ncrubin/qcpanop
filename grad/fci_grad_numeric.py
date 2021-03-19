import numpy as np
import scipy as sp
import openfermion as of
import fqe
import copy

from rhf_mo_grad import gradient_mo
from openfermionpyscf import run_pyscf
from openfermionpsi4 import run_psi4

from pyscf.lib.parameters import BOHR


def finite_diff_grad(molecule, step_size):
    """
    Perturb each atom x,y,z as x+h, x-h  etc to get force
    """
    de = np.zeros((len(molecule.geometry), 3))
    for atom_idx in range(len(molecule.geometry)):
        for xyz in range(3):
            new_geometry = copy.deepcopy(molecule.geometry)
            new_geometry[atom_idx][1][xyz] = step_size + new_geometry[atom_idx][1][xyz]
            pnew_molecule = of.MolecularData(geometry=new_geometry,
                                        basis=molecule.basis,
                                        charge=molecule.charge,
                                        multiplicity=molecule.multiplicity)
            pnew_molecule = run_pyscf(pnew_molecule, run_fci=True)
            pE = pnew_molecule.fci_energy

            new_geometry = copy.deepcopy(molecule.geometry)
            new_geometry[atom_idx][1][xyz] = new_geometry[atom_idx][1][xyz] - step_size
            mnew_molecule = of.MolecularData(geometry=new_geometry,
                                        basis=molecule.basis,
                                        charge=molecule.charge,
                                        multiplicity=molecule.multiplicity)
            mnew_molecule = run_pyscf(mnew_molecule, run_fci=True)
            mE = mnew_molecule.fci_energy

            # Remeber to multiply by Bohr to Angstrom conversion to get
            # Force in AU
            de[atom_idx, xyz] = (pE - mE) / (2 * step_size) * BOHR

    return de


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(
        verbose=0, atom='O   0.000000000000  -0.143225816552   0.000000000000;H  1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000',
        basis='sto-3g',
    )
    geometry = [['O', [0.000000000000, -0.143225816552, 0.000000000000]],
                ['H', [1.638036840407, 1.136548822547,  -0.000000000000]],
                ['H', [-1.638036840407, 1.136548822547,  -0.000000000000]]
                ]
    # mol = gto.M(
    #     verbose=0,
    #     atom='Li 0 0 0; H 0 0 1.5',
    #     basis='sto-3g',
    #     unit='Angstrom'
    # )
    # geometry = [['Li', [0, 0, 0]],
    #             ['H',  [0, 0, 1.5]]]

    mf = scf.RHF(mol)
    mf.kernel()

    molecule = of.MolecularData(geometry=geometry, basis=mol.basis,
                                charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_fci=True)
    nelec = molecule.n_electrons
    sz = 0
    oei, tei = molecule.get_integrals()

    w, v = sp.sparse.linalg.eigsh(of.get_sparse_operator(molecule.get_molecular_hamiltonian()).real)
    fqe_wf = fqe.from_cirq(v[:, 0].flatten(), 1.0E-12)
    print(w[0], molecule.fci_energy)
    fqe_wf.print_wfn()
    opdm, tpdm = fqe_wf.sector((nelec, sz)).get_openfermion_rdms()
    opdm = opdm.real
    tpdm = tpdm.real
    # opdm, tpdm = molecule.fci_one_rdm, molecule.fci_two_rdm
    # opdm = np.diag([1] * molecule.n_electrons + [0] * (molecule.n_qubits - molecule.n_electrons))
    # tpdm = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))

    # BE VERY CAREFUL HERE.  THE MO COEFFS must be those making OEI and TEI
    # OTHERWISE THINGS  WILL BE WRONG.
    de = gradient_mo(mol, mf.mo_coeff, oei, tei, opdm, tpdm)
    print(de)

    print(finite_diff_grad(molecule, 0.0001))

