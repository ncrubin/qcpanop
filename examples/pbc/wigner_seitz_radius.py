"""Calculate the Wigner-Seitz Radius for various crystals

(4/3) pi r_{s}^{3} = V / N

np.pow((3 V / (pi N)), 1/3)
"""

import numpy as np
from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic, Diamond

from pyscf.pbc import gto, scf
# divide by this quantity to get value in Bohr (atomic unit length)
from pyscf.lib.parameters import BOHR  # angstrom / bohr
from pyscf.pbc.tools import pyscf_ase


def wigner_seitz_radius(volume: float, n_electrons: int):
    """
    Calculate the Wigner-Seitz radius

    r_{s} = ((3 V / (4 pi N))^{1/3}

    :param volume: Volume of unit cell in Bohr^{3}
    :param n_electrons: number of electrons
    :return: float of wigner-seitz radius
    """
    return (3 / (4 * np.pi * (n_electrons / volume)))**(1/3)


def main():
    atom_dict = {'Li': BodyCenteredCubic('Li', latticeconstant=2.968),
                 'Na': BodyCenteredCubic('Na', latticeconstant=4.2906),
                 'K':  BodyCenteredCubic('K', latticeconstant=5.328),
                 'Rb': BodyCenteredCubic('Rb', latticeconstant=5.585),
                 'Cs': BodyCenteredCubic('Cs', latticeconstant=6.141)
                 }
    wsr = {}
    for atom_label, ase_atom in atom_dict.items():
        cell = gto.Cell()
        # cell.verbose = 5
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.a = ase_atom.cell
        # cell.basis = 'gth-szv'
        # cell.pseudo = 'gth-pade'
        # cell.build()
        print("Cell geometry")
        print(cell.lattice_vectors())
        print("atom vectors")
        print(cell.atom)

        # Omega = Cell volumes
        print()
        print("ASE Volume in Angstrom ", ase_atom.get_volume())
        print("Volume A^{3}", np.linalg.det(cell.a))
        print("Volume Bohr^{3}", np.linalg.det(cell.a / BOHR))
        print("PYSCF Volume Bohr^{3} ", cell.vol)
        nelectrons = sum(cell.nelec)

        print("Wigner-Seitz ", wigner_seitz_radius(20.121/ (BOHR**3),  1))
        exit()
        wsr[atom_label] = wigner_seitz_radius(cell.vol, 1)

    print(wsr)

    # ase_atom = Diamond('C')
    # cell = gto.Cell()
    # # cell.verbose = 5
    # cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    # cell.a = ase_atom.cell
    # # cell.basis = 'gth-szv'
    # # cell.pseudo = 'gth-pade'
    # # cell.build()
    # print("Cell geometry")
    # print(cell.lattice_vectors())
    # print("atom vectors")
    # print(cell.atom)

    # print(len(cell.atom))
    # print()

if __name__ == "__main__":
    main()