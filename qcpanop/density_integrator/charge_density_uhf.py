from pyscf import gto, scf, cc
from pyscf.tools import cubegen


def atom_calculator(*, atom_type: str, basis: str, qm_method: str,
                    nx_ngrid=80, ny_ngrid=80, nz_ngrid=80, box_dim=3.0):
    mol = gto.M()
    mol.atom = '{} 0 0 0'.format(atom_type)
    mol.basis = basis
    mol.build()

    if qm_method == 'hf':
        mf = scf.UHF(mol)
        mf.init_guess_breaksym = 'True'
        mf.verbose = 4
        mf.kernel()
        rho = cubegen.density(mol, 'mol_density.cube',
                        mf.make_rdm1()[0] + mf.make_rdm1()[1],
                              nx=nx_ngrid, ny=ny_ngrid, nz=nz_ngrid,
                              margin=box_dim
                              )  # makes total density
    else:
        raise ValueError("{} is not a valid qm method".format(qm_method))

    print(rho.shape)




if __name__ == "__main__":
    atom_calculator(atom_type='C', qm_method='hf', basis='cc-pvtz',
                    nx_ngrid=100, ny_ngrid=100, nz_ngrid=100)