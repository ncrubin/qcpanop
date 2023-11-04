"""
Exploring the wedge product representation
"""
import itertools
from math import prod
import numpy as np
import openfermion as of
from itertools import combinations
import fqe
from math import factorial
from scipy.special import comb

def kdelta(i, j):
    if np.isclose(i, j):
        return 1.
    else:
        return 0.

def int_to_ladder_op(string_int, width, ket=True):
    bstring = list(map(int, np.binary_repr(string_int, width=width)))
    bstring_fop = []
    for idx, ii in enumerate(bstring):
        if ii == 0:
            bstring_fop.append(of.FermionOperator(((idx, 0), (idx, 1))))
        else:
            if ket:
                bstring_fop.append(of.FermionOperator(((idx, 1))))
            else:
                bstring_fop.append(of.FermionOperator(((idx, 0))))
    return prod(bstring_fop)

def int_to_ladder_op_new_order(string_int, width, flag=None):
    bstring = list(map(int, np.binary_repr(string_int, width=width)))
    bstring_fop = []
    zero_indices = []
    for idx, ii in enumerate(bstring):
        if ii == 0:
            # bstring_fop.append(of.FermionOperator(((idx, 0), (idx, 1))))
            zero_indices.append(idx)
        else:
            bstring_fop.append(of.FermionOperator(((idx, 1))))

    if len(bstring_fop) == 0:
        return of.FermionOperator(()), zero_indices
    else:
        return prod(bstring_fop), zero_indices

def get_zero_ops(indices):
    zero_bstrings = []
    for ii in indices:
        zero_bstrings.append(of.FermionOperator(((ii, 0), (ii, 1))))
    if len(indices) == 0:
        return of.FermionOperator(())
    else:
        return prod(zero_bstrings)

def change_raise_lowering(fop, position):
    assert len(fop.terms) == 1
    new_fop = []
    terms = list(fop.terms.keys())[0]
    phase = 1
    for key in terms:
        if key[0] == position and key[1] == 1:
            new_fop.append((position, 0))
            phase = (-1)**(len(new_fop[:-1]))
        elif key[0] == position and key[1] == 0:
            new_fop.append((position, 1))
            phase = (-1)**(len(new_fop[:-1]))
        else:
            new_fop.append(key)
    return of.FermionOperator(tuple(new_fop)), phase

def build_creation_string(jj):
    return of.FermionOperator(([(xx, 1) for xx in map(int, jj)]))

def identity_representation():
    m = 3
    num_electrons = 1
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    # fqe_wf = fqe.Wavefunction([[num_electrons, sz, m], [num_electrons, -sz, m]])
    # fqe_wf.set_wfn(strategy='random')
    # fqe_wf.print_wfn()
    # fqe_data = fqe_wf.sector((num_electrons, sz))
    # opdm, tpdm = fqe_data.get_openfermion_rdms()
    # wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    # rho = wf @ wf.conj().T

    non_zero_idx = []
    compliment_idx = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)
        else:
            compliment_idx.append(jj)


    fermion_op_identity = of.FermionOperator()
    for jj in range(2**num_qubits):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        jj_ket = np.binary_repr(jj, width=num_qubits)
        jja_ket = jj_ket[::2]
        jjb_ket = jj_ket[1::2]
        sz_jj_ket = jja_ket.count('1') - jjb_ket.count('1')

        for kk in range(2**num_qubits):
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            kk_ket = np.binary_repr(kk, width=num_qubits)
            kka_ket = kk_ket[::2]
            kkb_ket = kk_ket[1::2]
            sz_kk_ket = kka_ket.count('1') - kkb_ket.count('1')


            if jj_ket.count('1') == 1 and kk_ket.count('1') == 1 and jj_ket == kk_ket:
                print(bstring_fop_ket * bstring_fop_bra)
                fermion_op_identity += bstring_fop_ket * bstring_fop_bra * zero_fop
    
    identity_mat = of.get_sparse_operator(fermion_op_identity).todense()

    compliment_mat = identity_mat.copy()
    compliment_mat = compliment_mat[:, compliment_idx]
    compliment_mat = compliment_mat[compliment_idx, :]
    assert np.allclose(compliment_mat, 0)

    identity_mat = identity_mat[:, non_zero_idx]
    identity_mat = identity_mat[non_zero_idx, :]
    assert np.allclose(identity_mat.real, np.eye(num_qubits))

def one_body_representation_one_particle_space():
    np.set_printoptions(linewidth=500)
    np.random.seed(10)
    m = 3
    num_electrons = 1
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    # fqe_wf = fqe.Wavefunction([[num_electrons, sz, m], [num_electrons, -sz, m]])
    # fqe_wf.set_wfn(strategy='random')
    # fqe_wf.print_wfn()
    # fqe_data = fqe_wf.sector((num_electrons, sz))
    # opdm, tpdm = fqe_data.get_openfermion_rdms()
    # wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    # rho = wf @ wf.conj().T

    A = np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits)) \
         + 1j * np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits))
    A = A + A.conj().T
    assert of.is_hermitian(A)
    A_fop = of.FermionOperator()
    for i, j in itertools.product(range(num_qubits), repeat=2):
        A_fop += of.FermionOperator(((i, 1), (j, 0)), coefficient=A[i, j])
    assert of.is_hermitian(A_fop)

    A_fop_mat = of.get_sparse_operator(A_fop).todense()

    non_zero_idx = []
    compliment_idx = []
    one_idx = []
    two_idx = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)
        else:
            compliment_idx.append(jj)

        if np.binary_repr(jj).count('1') == 1:
            one_idx.append(jj)
        if np.binary_repr(jj).count('1') == 2:
            two_idx.append(jj)
        
    
    A_fop_mat = of.get_sparse_operator(A_fop).todense()
    A_fop_mat = A_fop_mat[:, one_idx]
    A_fop_mat = A_fop_mat[one_idx, :]
    print(A_fop_mat)
    print()
    print(A[::-1, ::-1])
    assert np.allclose(A_fop_mat, A[::-1, ::-1])

    for ii in one_idx:
        print(np.binary_repr(ii, width=num_qubits))

    print("Raising")
    vac = np.zeros((2**num_qubits, 1), dtype=np.complex128)
    vac[0, 0] = 1
    res = of.get_sparse_operator(of.FermionOperator((0, 1)), n_qubits=num_qubits).todense() @ vac
    for ii in range(2**num_qubits):
        if np.isclose(res[ii, 0], 1):
            print(np.binary_repr(ii, width=num_qubits))
            print(int(np.binary_repr(ii, width=num_qubits), 2))

def one_body_representation_two_particle_space():
    np.set_printoptions(linewidth=500)
    np.random.seed(10)
    m = 5
    num_electrons = 1
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m

    A = np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits)) \
         + 1j * np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits))
    A = A + A.conj().T
    assert of.is_hermitian(A)
    # A = np.eye(num_qubits)
    A_fop = of.FermionOperator()
    for i, j in itertools.product(range(num_qubits), repeat=2):
        A_fop += of.FermionOperator(((i, 1), (j, 0)), coefficient=A[i, j])
    assert of.is_hermitian(A_fop)

    A_fop_mat = of.get_sparse_operator(A_fop).todense()

    non_zero_idx = []
    compliment_idx = []
    one_idx = []
    two_idx = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)
        else:
            compliment_idx.append(jj)

        if np.binary_repr(jj).count('1') == 1:
            one_idx.append(jj)
        if np.binary_repr(jj).count('1') == 2:
            two_idx.append(jj)
        
    
    A_fop_mat = of.get_sparse_operator(A_fop).todense()
    for jj in two_idx:
        jj_ket = np.binary_repr(jj, width=num_qubits)
        jj_idx = np.argwhere(list(map(int, jj_ket))).ravel()
        for kk in two_idx:
            kk_ket = np.binary_repr(kk, width=num_qubits)
            kk_idx = np.argwhere(list(map(int, kk_ket))).ravel()

            if jj == kk and np.binary_repr(jj).count('1') == 2:
                print(A_fop_mat[jj, kk], np.sum([A[xx, xx] for xx in jj_idx]))
                assert np.isclose(A_fop_mat[jj, kk], np.sum([A[xx, xx] for xx in jj_idx]))
            elif np.binary_repr((jj ^ kk)).count('1') == 2 and jj_ket.count('1') == 2 and kk_ket.count('1') == 2:
                pq_idx = np.argwhere(list(map(int, np.binary_repr((jj ^ kk), width=num_qubits)))).ravel()
                test_vac = np.binary_repr((jj ^ kk), width=num_qubits) 
                assert np.isclose(abs(A_fop_mat[jj, kk]), abs(A[pq_idx[0], pq_idx[1]]))
                i, k = jj_idx
                j, l = kk_idx
                coeff = A[i, j] * kdelta(k, l) - A[k, j] * kdelta(i, l) - A[i, l] * kdelta(k, j) + A[k, l] * kdelta(i, j)
                assert np.isclose(A_fop_mat[jj, kk], coeff)

    A_coeffs = np.einsum("ij,kl->ikjl", A, np.eye(num_qubits))

    print(np.einsum('mijm', A_coeffs))
    print()
    print(A)
    assert np.allclose(A, np.einsum('mijm', A_coeffs))

def random_one_state_wedged_to_two():
    np.set_printoptions(linewidth=500)
    np.random.seed(10)
    m = 3
    num_electrons = 1
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m], [num_electrons, -sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T



if __name__ == "__main__":
    identity_representation()
    # one_body_representation_two_particle_space()