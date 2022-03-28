import numpy as np
from pyscf import gto, scf, fci
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from doci import get_mo_via_cas, pyscf_to_fqe_wf

def example1():
    dim = 10
    A = np.array([[2, -12], [1, -5]], dtype=float)
    A = np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.T
    x0 = np.random.randn(dim).T
    w, v = np.linalg.eigh(A)
    print(w)

    residual = np.inf
    x = x0.copy()
    x_old = x0.copy()
    iter = 0
    iter_max = 100
    while residual > 1.0E-4 and iter < iter_max:
        x = A @ x
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        if iter % 10 == 0:
            print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1
    print("correct sign is lam/x[0,0]", np.sign(lam/x[0]))


def example2():
    mol = gto.M()
    mol.atom = 'Li 0 0 0; H 0 0 1.6'
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    # Get MO integrals
    le_ham, ecore, h1, eri = get_mo_via_cas(mf)
    cisolver = fci.FCI(mf)
    fci_e, fci_civec = cisolver.kernel()
    fci_fqe_wf = pyscf_to_fqe_wf(fci_civec, pyscf_mf=mf)
    ele_ham, ecore, h1, eri = get_mo_via_cas(mf)
    assert np.isclose(fci_e, fci_fqe_wf.expectationValue(ele_ham).real)

    # Get Hamiltonian matrix for power method
    spin_oei, spin_tei = spinorb_from_spatial(h1, eri)
    aspin_tei = 0.25 * (spin_tei - np.einsum('ijlk', spin_tei))
    mol_ham = of.InteractionOperator(ecore, spin_oei, aspin_tei)
    print("gettings sparse operator")
    A = of.get_sparse_operator(mol_ham).toarray()
    C = np.min(A.diagonal())
    C_eye = np.eye(A.shape[0]) * C  # this is so we can shift the energy such that the diagonals are positive.
    w, v = np.linalg.eigh(A)
    print("Checking energy")
    assert np.isclose(w[0], fci_e)
    print("GS Energy ", w[0])
    print("Maximal eig ", w[-1])

    # get intial state
    x0 = of.jw_hartree_fock_state(mf.mol.nelectron, mf.mo_coeff.shape[1] * 2)

    # perform power method iteration
    residual = np.inf
    x = x0.copy()
    x_old = x0.copy()
    iter = 0
    iter_max = 100
    K = -C_eye - A
    print(np.diagonal(K))
    exit()
    while iter < iter_max:
        x = K @ x
        lam = np.linalg.norm(x)
        x /= np.linalg.norm(x)
        residual = np.abs(x.T @ x_old)
        x_old = x.copy()
        if iter % 10 == 0:
            print(iter, residual, lam, x.T @ A @ x / (x.T @ x))
        iter += 1


if __name__ == "__main__":
    example2()
