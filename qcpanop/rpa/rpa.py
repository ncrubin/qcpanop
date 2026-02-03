"""

random phase approximation correlation energy

"""

import warnings
import numpy as np
import scipy

from pyscf import ao2mo

from openfermion.chem.molecular_data import spinorb_from_spatial

class rpa:

    def __init__(self, mol, mf):
        """
        initialize rpa class

        :params mol: pyscf molecule object
        :params mf: pyscf mean field object

        """

        C = mf.mo_coeff
       
        # one-electron integrals 
        h_ao = mf.get_hcore()
        h = C.conj().T @ h_ao @ C

        occ = mf.mo_occ
        nele = int(sum(occ))
        nocc = nele // 2
        norbs = h.shape[0]
        nv = 2 * (norbs - nocc)
        no = 2 * nocc

        # two-electron integrals
        tmp = ao2mo.kernel(mol, C)
        tmp = ao2mo.restore(1, tmp, C.shape[1])

        J = np.einsum('pqkk->pq', tmp[:, :, :nocc, :nocc])
        K = np.einsum('pkkq->pq', tmp[:, :nocc, :nocc, :])
        f = h + 2 * J - K

        self.f, tmp = spinorb_from_spatial(f, tmp)
        self.g = tmp.transpose(0, 2, 1, 3) # physicists' notation

        self.no = no
        self.nv = nv
        self.ov = no * nv
        self.o = slice(None, self.no)
        self.v = slice(self.no, None)

        kd = np.eye(no+nv)

        # B(ia, jb) = <ij|ab>
        self.B = self.g[self.o, self.o, self.v, self.v].transpose(0, 2, 1, 3).reshape(self.ov, self.ov)

        # A(ia, jb) = <ib|aj> + f(ab)dij - f(ij)dab
        tmp = self.g[self.o, self.v, self.v, self.o].transpose(0, 2, 3, 1)
        tmp += np.einsum('ab,ij->iajb', self.f[self.v, self.v], kd[self.o, self.o])
        tmp -= np.einsum('ij,ab->iajb', self.f[self.o, self.o], kd[self.v, self.v])
        self.A = tmp.reshape(self.ov, self.ov)

    def correlation_energy(self):
        """
        rpa correlation energy from solving rpa eigenvalue problem
        """

        print("")
        print("    ==> RPA Correlation Energy from 1/2 Tr(w-A) <==")
        print("")

        # (A+B)(A-B)(X-Y) = E^2(X-y)
        tmp = (self.A - self.B) @ (self.A + self.B)
        eig, vec = np.linalg.eig(tmp)
        rpa_eig = eig**0.5

        rpa_correlation_energy = 0.5 * np.sum(rpa_eig) - 0.5 * np.einsum('ii->', self.A)

        print("    RPA Correlation Energy: {: 20.12f}".format(rpa_correlation_energy))
        print("")

        return rpa_correlation_energy

    def ricatti_solver(self, maxiter = 50, r_convergence = 1e-6, e_convergence = 1e-8):

        from scipy.linalg import solve_continuous_lyapunov
        T = np.zeros((self.ov, self.ov))

        print("")
        print("    ==> RPA Ricatti Equation <==")
        print("")
        print("     Iter               Energy                 |dE|                  |R|")

        old_energy = 0
        for it in range(maxiter):
            R = self.B + self.A @ T + T @ self.A + T @ self.B @ T
            res_norm = np.linalg.norm(R, ord='fro')

            M = self.A + self.B @ T
            # solve M^T Δ + Δ M = -R
            Delta = solve_continuous_lyapunov(M.T, -R)
            T = T + Delta

            current_energy = 0.5 * np.einsum('ij,ij->', self.B, T)
            delta_e = np.abs(old_energy - current_energy)

            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(it, current_energy, delta_e, res_norm))

            if res_norm < r_convergence and delta_e < e_convergence:
                break
            else:
                old_energy = current_energy

        else:
            raise ValueError("Ricatti iterations did not converge")

        print("")
        print("    RPA Correlation Energy: {: 20.12f}".format(current_energy))
        print("")


            
        return current_energy

