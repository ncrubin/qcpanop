"""
Implementation of GFMC
"""
from typing import List, Tuple
from itertools import product
import numpy as np
from scipy.sparse import csc_matrix
from pyscf import gto, scf, fci

import openfermion as of


from openfermion.chem.molecular_data import spinorb_from_spatial

import matplotlib.pyplot as plt


class PsiTrial:

    def __init__(self, psi_t: np.ndarray):
        """Sample psi_t class. Technically Psi_t needs to have efficient access
        to <n|psi_T> and <n|H|psi_T>. For learning purposes we will just input a
        vector |psi_T> that is 2**N elements"""
        self.psi_t = psi_t
        self.dim = int(max(psi_t.shape))

    def generate_Hpsi_t(self, hamiltonian):
        self.h_psi_t = hamiltonian @ self.psi_t

    def get_n_psit(self, n):
        return self.psi_t[n, 0]

    def get_n_h_psit(self, n):
        return self.h_psi_t[n, 0]

    def local_e(self, n):
        return self.get_n_h_psit(n) / self.get_n_psit(n)


class DidacticGFMC:

    def __init__(self, qubit_op_hamiltonian: of.QubitOperator,
                 num_walkers: int, batch_size: int,
                 psi_trial=None):
        self.hamiltonian = qubit_op_hamiltonian
        self.sparse_ham = of.get_sparse_operator(qubit_op_hamiltonian)
        self.num_qubits = of.utils.count_qubits(self.hamiltonian)

        self.num_walkers = num_walkers
        self.total_steps = 20
        self.batch_size = batch_size
        self.num_batches = self.num_walkers // self.batch_size
        self.psi_trial = psi_trial
        self.current_energy_est = None
        self.prop_matrix = None

        # just for slow version of GFMC
        identity_matrix = csc_matrix(([1]*(2**self.num_qubits), (np.arange(2**self.num_qubits),
                                                                 np.arange(2**self.num_qubits))), dtype=np.complex128)
        self.identity_matrix = identity_matrix


    def get_connected_bitstrings(self):
        # build connections for H_{x',x}
        self.connected_bitstrings = {}
        for ii in range(2 ** self.num_qubits):
            self.connected_bitstrings[ii] = self.prop_matrix.getrow(ii).indices

    def build_propagator(self):
        if self.current_energy_est is None:
            self.current_energy_est = 0.
        self.prop_matrix = 10 * self.identity_matrix - self.sparse_ham  #+ self.current_energy_est * self.identity_matrix
        self.get_connected_bitstrings()

    def build_weight(self, hilbert_space_basis_index: int) -> float:
        """
        Compute b_{x} = sum_{x'}G_{x',x}

        :param hilbert_space_basis_index: x-in the equation above
        :return: float b_{x}
        """
        return sum(abs(self.prop_matrix.getcol(hilbert_space_basis_index).data))

    def get_transition_prob(self, i: int, j: int) -> float:
        """
        Compute P_{x',x} = G_{x',x}/b_{x}

        from_idx = x,  to_idx x '

        :param from_idx: x in above equation
        :param to_idx: x' in above equation
        :return: float --conditional probability to go from x to x'
        """
        if not np.isclose(self.build_weight(j), 0):
            return abs(self.prop_matrix[i, j].real) / self.build_weight(j)
        else:
            return 0.

    def transition_walker(self, walker_i, walker_w) -> Tuple[int, float]:
        """Transition walker to the next iteration"""
        transition_probs = [self.get_transition_prob(xx, walker_i) for xx in
                            self.connected_bitstrings[walker_i]]
        assert np.isclose(sum(transition_probs), 1)
        new_walker_position = np.random.choice(
            self.connected_bitstrings[walker_i], p=transition_probs)
        # update weight and walker
        assert np.sign(self.prop_matrix[new_walker_position, walker_i]) >= 0
        walker_weight = walker_w * self.build_weight(walker_i) * np.sign(self.prop_matrix[new_walker_position, walker_i])
        return new_walker_position, walker_weight

    def energy_estimation(self, walkers_idx: List[int]=None, weights_idx: List[int]=None) -> Tuple[float, float]:
        """
        :param walkers_idx: List of walker integers
        :param weights_idx: iteratable with weights for each walker in order.
        """
        if walkers_idx is None:
            walkers_idx = self.walker_idx
        if weights_idx is None:
            weights_idx = self.walker_weights

        average_energy_of_batch = []
        for bidx in range(self.num_batches):
            walkers_idx_batch = walkers_idx[bidx * self.batch_size:(bidx + 1) * self.batch_size]
            weights_idx_batch = weights_idx[bidx * self.batch_size:(bidx + 1) * self.batch_size]
            trial_energy = self.batch_energy_estimation(walkers_idx_batch, weights_idx_batch)
            average_energy_of_batch.append(trial_energy)
        self.current_energy_est = np.mean(average_energy_of_batch)
        return np.mean(average_energy_of_batch).real, np.std(average_energy_of_batch)

    def batch_energy_estimation(self, walkers_idx: List[int], weights_idx: List[int]) -> float:
        """
        :param walkers_idx: List of walker integers
        :param weights_idx: iteratable with weights for each walker in order.
        """
        numerator = 0
        denominator = 0
        # obviously this can be vectorized
        for walker_index, (widx, hidx) in enumerate(zip(walkers_idx, weights_idx)):
            local_energy = self.psi_trial.get_n_h_psit(widx) / self.psi_trial.get_n_psit(widx)
            numerator += local_energy * hidx  * self.psi_trial.psi_t[widx] * self.walker_init_psi_t[walker_index]
            denominator += hidx  * self.psi_trial.psi_t[widx] * self.walker_init_psi_t[walker_index]
        return float(numerator.real / denominator.real)


    def init_walkers(self, psi_trial: PsiTrial):
        """Initialize to number presering basis"""
        self.psi_trial = psi_trial
        walker_idx = np.random.choice(list(range(self.psi_trial.dim)),
                                      size=self.num_walkers, p=[1/self.psi_trial.dim] * self.psi_trial.dim)
        walker_weights = np.ones(len(walker_idx))
        walker_init_psi_t = np.array([self.psi_trial.psi_t[xx, 0] for xx in walker_idx])
        self.walker_idx = walker_idx
        self.walker_weights =  walker_weights
        self.walker_init_psi_t = walker_init_psi_t

    def evolve_walkers(self, walkers_idx: List[int] = None, weights_idx: List[int] = None):
        """
        Evolve walkers based on Markov chain probability

        :param walkers_idx:
        :param weights_idx:
        :return:
        """
        if walkers_idx is None:
            walkers_idx = self.walker_idx
        if weights_idx is None:
            weights_idx = self.walker_weights
        # build_prop_matrix
        self.build_propagator()
        new_walkers = []
        new_weights = []
        for walker, weight in zip(walkers_idx, weights_idx):
            new_walker, new_weight = self.transition_walker(walker, weight)
            new_walkers.append(new_walker)
            new_weights.append(new_weight)
        return new_walkers, new_weights

    def calc_avg_sign(self, walkers_idx: List[int] = None, weights_idx: List[int] = None):
        if walkers_idx is None:
            walkers_idx = self.walker_idx
        if weights_idx is None:
            weights_idx = self.walker_weights
        return np.sum(walker_weights) / np.sum(np.abs(walker_weights))



def print_wfn(wf, nqubits):
    for ii in range(2 ** nqubits):
        if not np.isclose(wf[ii, 0], 0):
            print(np.binary_repr(ii, width=nqubits), wf[ii, 0])


def gfmc_example1():
    # np.random.seed(10)
    # get Heisnberg lattice
    dim = 4
    hamiltonian = of.QubitOperator((), coefficient=0)
    J = 2
    for site_idx in range(dim - 1):
        hamiltonian += of.QubitOperator(((site_idx, 'X'), (site_idx + 1, 'X')),
                                        coefficient=J)
        hamiltonian += of.QubitOperator(((site_idx, 'Y'), (site_idx + 1, 'Y')),
                                        coefficient=J)
    print(hamiltonian)

    gfmc_inst = DidacticGFMC(hamiltonian, 100_000, 100_000)
    A = gfmc_inst.sparse_ham.toarray().real
    w, v = np.linalg.eigh(A)

    print(v[:, [0]])
    exit()

    from scipy.linalg import expm
    G = expm(-A)
    sign_indicator = np.zeros_like(G)
    fixed_node_H = np.zeros_like(G)
    H = A

    psi_t = abs(v[:, [0]]) + 0.01 * abs(np.random.randn(2 ** dim).reshape((-1, 1)))
    psi_t /= np.linalg.norm(psi_t)
    for i, j in product(range(2**dim), repeat=2):
        sign_indicator[i, j] = G[i, j] * psi_t[i, 0] * psi_t[j, 0]
        if H[i, j] == 4:
            print(psi_t[i, 0], psi_t[j, 0])
        if H[i, j] * psi_t[i, 0] * psi_t[j, 0] < 0:
            fixed_node_H[i, j] = H[i, j]
        if i == j:
            v_sf = 0
            for rp in range(2**dim):
                if H[i, rp] * psi_t[i, 0] * psi_t[rp, 0] > 0:
                    v_sf += H[i, rp] * psi_t[rp, 0] / psi_t[i]
            fixed_node_H[i, j] = H[i, j] + v_sf

    print(fixed_node_H)
    G_fn = expm(-fixed_node_H)
    import matplotlib.pyplot as plt
    plt.imshow(G_fn.real)
    plt.colorbar()
    plt.show()
    exit()

    w, v = np.linalg.eigh(A)
    I = np.eye(2 ** dim)
    psi_t = abs(v[:, [0]]) + 0.01 * abs(np.random.randn(2 ** dim).reshape((-1, 1)))
    psi_t /= np.linalg.norm(psi_t)
    assert np.isclose(psi_t.conj().T @ psi_t, 1)
    psi_g = psi_t
    sigma = psi_g.conj().T @ A @ psi_g
    G_vals = []
    power_steps = 4
    pwer_energy_steps = []
    for _ in range(power_steps):
        G_vals.append(abs(sigma) * I - A)
        assert np.alltrue(G_vals[-1] >= 0)
        psi_g = (abs(sigma) * I - A) @ psi_g
        # psi_g /= np.linalg.norm(psi_g)
        psi_g_normed = psi_g.copy()
        psi_g_normed /= np.linalg.norm(psi_g_normed)
        eig_est = psi_g_normed.conj().T @ A @ psi_g_normed
        pwer_energy_steps.append(eig_est)
        print(eig_est, eig_est - w[0], np.linalg.norm(psi_g))
        sigmaI = eig_est

    psi_trial = PsiTrial(psi_t=psi_t)
    psi_trial.generate_Hpsi_t(A)
    gfmc_inst.init_walkers(psi_trial)
    E_vmc_psi_t = gfmc_inst.batch_energy_estimation(gfmc_inst.walker_idx,
                                                    gfmc_inst.walker_weights)
    print("Initial energy ", E_vmc_psi_t)
    print("Exact initial energy ", (psi_t.conj().T @ A @ psi_t)[0, 0].real)
    print("Exact ground state energy ", w[0])
    gfmc_inst.build_propagator()
    gfmc_inst.get_connected_bitstrings()
    for iteration in range(6):
        new_walkers, new_walker_weights = gfmc_inst.evolve_walkers()
        gfmc_inst.walker_idx = new_walkers
        gfmc_inst.walker_weights = new_walker_weights
        energy_est = gfmc_inst.batch_energy_estimation(gfmc_inst.walker_idx,
                                                       gfmc_inst.walker_weights)
        print("{}\t{:5.10f}\t{:5.10f}\t{:5.10f}".format(
            iteration, energy_est, np.max(gfmc_inst.walker_weights),
            np.min(gfmc_inst.walker_weights)
        )
        )



if __name__ == "__main__":
    gfmc_example1()
