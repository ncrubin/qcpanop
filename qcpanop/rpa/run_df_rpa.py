from qcpanop.rpa.rpa import rpa

from pyscf import gto, scf

def main():

    mol = gto.M(atom='O            0.000000000000     0.000000000000    -0.068516219320; \
                      H            0.000000000000    -0.790689573744     0.543701060715; \
                      H            0.000000000000     0.790689573744     0.543701060715',
                basis='cc-pvdz')

    mf = scf.RHF(mol).density_fit().run()

    myrpa = rpa(mol, mf, use_df = True)

    # correlation energy from imaginary frequency integration
    ec = myrpa.correlation_energy()

if __name__ == "__main__":
    main()
