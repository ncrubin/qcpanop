from qcpanop.rpa.rpa import rpa

from pyscf import gto, scf

def main():

    mol = gto.M(atom='O            0.000000000000     0.000000000000    -0.068516219320; \
                      H            0.000000000000    -0.790689573744     0.543701060715; \
                      H            0.000000000000     0.790689573744     0.543701060715',
                basis='cc-pvdz')

    mf = scf.RHF(mol).run()

    myrpa = rpa(mol, mf)

    # correlation energy from solving the RPA eigenvalue problem
    ec = myrpa.correlation_energy()

    # correlation energy from solving the RPA Ricatti equation
    ec = myrpa.ricatti_solver()

if __name__ == "__main__":
    main()
