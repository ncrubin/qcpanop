"""

random phase approximation correlation energy

"""

import numpy as np

from pyscf import ao2mo, df, lib

from openfermion.chem.molecular_data import spinorb_from_spatial

from scipy.special import roots_legendre

def get_imag_freq_grid(n_points, scaling_factor=1.0):
    """
    imaginary frequency grid for rpa quadrature
    """

    # 1. Get standard Gauss-Legendre roots (t) and weights (w) defined on [-1, 1]
    # 'leggauss' returns roots and weights.
    t, w = np.polynomial.legendre.leggauss(n_points)
    
    # 2. Transform roots from [-1, 1] to [0, inf)
    # The variable change is: omega = x0 * (1 + t) / (1 - t)
    denominator = 1.0 - t
    freqs = scaling_factor * (1.0 + t) / denominator
    
    # 3. Transform weights using the Jacobian
    # d(omega)/dt = 2 * x0 / (1 - t)^2
    jacobian = (2.0 * scaling_factor) / (denominator**2)
    weights = w * jacobian
    
    # Return as a list of pairs for easy looping
    return zip(freqs, weights)

class rpa:

    def __init__(self, mol, mf, use_df = False):
        """
        initialize rpa class

        :param mol: pyscf molecule object
        :param mf: pyscf mean field object

        """

        self.use_df = use_df
        self.mol = mol
        self.mf = mf

    def build_AB(self):
        """
        build RPA A and B matrices
        """

        C = self.mf.mo_coeff
    
        # one-electron integrals 
        h_ao = self.mf.get_hcore()
        h = C.conj().T @ h_ao @ C

        occ = self.mf.mo_occ
        nele = int(sum(occ))
        nocc = nele // 2
        norbs = h.shape[0]
        nv = 2 * (norbs - nocc)
        no = 2 * nocc

        self.no = no
        self.nv = nv
        self.ov = no * nv
        self.o = slice(None, self.no)
        self.v = slice(self.no, None)

        # two-electron integrals
        tmp = ao2mo.kernel(self.mol, C)
        tmp = ao2mo.restore(1, tmp, C.shape[1])

        J = np.einsum('pqkk->pq', tmp[:, :, :nocc, :nocc])
        K = np.einsum('pkkq->pq', tmp[:, :nocc, :nocc, :])
        f = h + 2 * J - K

        f, tmp = spinorb_from_spatial(f, tmp)
        g = tmp.transpose(0, 2, 1, 3) # physicists' notation

        kd = np.eye(no+nv)

        # B(ia, jb) = <ij|ab>
        self.B = g[self.o, self.o, self.v, self.v].transpose(0, 2, 1, 3).reshape(self.ov, self.ov)

        # A(ia, jb) = <ib|aj> + f(ab)dij - f(ij)dab
        tmp = g[self.o, self.v, self.v, self.o].transpose(0, 2, 3, 1)
        tmp += np.einsum('ab,ij->iajb', f[self.v, self.v], kd[self.o, self.o])
        tmp -= np.einsum('ij,ab->iajb', f[self.o, self.o], kd[self.v, self.v])
        self.A = tmp.reshape(self.ov, self.ov)


    def correlation_energy(self, npts = 40):
        """
        rpa correlation energy from imaginary frequency integration

        :param npts: number of quadrature points

        :return correlation energy
        """

        if not self.use_df:
            raise Exception("RPA imaginary frequency integration requires density fitting")

        print("")
        print("    ==> RPA Correlation Energy from Response Function <==")
        print("")

        C = self.mf.mo_coeff

        # one-electron integrals 
        h_ao = self.mf.get_hcore()
        h = C.conj().T @ h_ao @ C

        occ = self.mf.mo_occ
        nele = int(sum(occ))
        nocc = nele // 2
        norbs = h.shape[0]

        # three-index integrals       
        B_pq_ao = lib.unpack_tril(self.mf.with_df._cderi)
        tmp = np.dot(B_pq_ao, C)
        B_pq = np.einsum('up,Auv->Apv', C, tmp)
        
        # Jpq = sum_i (pq|A)(A|ii)
        tmp = np.einsum('Aii->A', B_pq[:, :nocc, :nocc])
        J = np.einsum('Apq,A->pq', B_pq, tmp)

        # Kpq = sum_i (pi|A)(A|qi)
        K = np.einsum('Api,Aqi->pq', B_pq[:, :, :nocc], B_pq[:, :, :nocc])
        
        # fock matrix
        f = h + 2 * J - K

        # flattened matrix of orbital energy differences
        eps = np.diag(f)
        eps_i = eps[:nocc]
        eps_a = eps[nocc:]
        eps_ai = (eps_a[:, None] - eps_i[None, :]).flatten()
        #eps_ai = (eps_a[None, :] - eps_i[:, None]).flatten()

        # vo part of three-index integrals
        B_ai = B_pq[:, nocc:, :nocc].reshape(B_pq.shape[0], nocc * (norbs - nocc))
        #B_ai = B_pq[:, :nocc, nocc:].reshape(B_pq.shape[0], nocc * (norbs - nocc))

        # quadrature grid
        grid = get_imag_freq_grid(npts, scaling_factor = eps_ai.min())

        # check mp2 energy 
        #g = np.einsum('Ap,Aq->pq', B_ai, B_ai)
        #e2 = 0.0
        #for i in range(nocc):
        #    for j in range(nocc):
        #        for a in range(norbs - nocc):
        #            for b in range(norbs - nocc):
        #                denom = eps_i[i] + eps_i[j] - eps_a[a] - eps_a[b]
        #                num = g[a*nocc+i, b*nocc+j] * (2 * g[a*nocc+i, b*nocc+j] - g[a*nocc+j, b*nocc+i])
        #                e2 += num / denom
        #print(e2)

        # imaginary frequency integration
        rpa_correlation_energy = 0.0

        for omega, weight in grid:
            
            fac = 4.0 * eps_ai / (eps_ai**2 + omega**2)

            Q = B_ai @ (fac[:, None] * B_ai.T)
            trQ = np.sum(np.diag(Q))

            # 1 + Q
            Q += np.eye(Q.shape[0])
            
            eigvals = np.linalg.eigvalsh(Q)
            
            # Tr[ln(1 + Q) - Q]
            val = np.sum(np.log(eigvals)) - trQ

            rpa_correlation_energy += weight * val

        rpa_correlation_energy /= 2.0 * np.pi 

        print("    RPA Correlation Energy: {: 20.12f}".format(rpa_correlation_energy))
        print("")

        return rpa_correlation_energy

    def correlation_energy_from_eigensolver(self):
        """
        rpa correlation energy from solving rpa eigenvalue problem

        :return correlation energy, 0.5 Tr(w - A)
        """

        if self.use_df:
            raise Exception("RPA eigensolver does does not work with density fitting")

        self.build_AB()

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
        """
        solve B + A@T + T@A + T@B@T = 0
        
        :param maxiter: maximum number of iterations
        :param r_convergence: convergence in residual
        :param e_convergence: convergence in correlation energy, 0.5 Tr(B@T)

        :return correlation energy, 0.5 Tr(B@T)
        """

        if self.use_df:
            raise Exception("Ricatti solver does not work with density fitting")

        self.build_AB()

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

            # solve M^T dT + dT M = -R
            dT = solve_continuous_lyapunov(M.T, -R)
            T = T + dT

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
