import numpy as np
from pyscf.fci.cistring import make_strings
import fqe

def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = [x for x in make_strings(norb_list, nelec[0])]
    n_beta_strings = [x for x in make_strings(norb_list, nelec[1])]

    fqe_wf_ci = fqe.Wavefunction([[sum(nelec), nelec[0] - nelec[1], norbs]])
    fqe_data_ci = fqe_wf_ci.sector((sum(nelec), nelec[0] - nelec[1]))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()))
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            # if np.abs(civec[paidx, pbidx]) > 1.0E-3:
            #     print(np.binary_repr(pyscf_alpha_idx, width=10), np.binary_repr(pyscf_beta_idx, width=10), civec[paidx, pbidx])
            fqe_orderd_coeff[fqe_graph_ci.index_alpha(
                pyscf_alpha_idx), fqe_graph_ci.index_beta(pyscf_beta_idx)] = \
                pyscf_cimat[paidx, pbidx]

    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci