"""
Algebraically express a density matrix
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


def main():
    m = 3
    psi = np.random.randn(2**m) + 1j * np.random.randn(2**m)
    psi /= np.linalg.norm(psi)
    assert np.isclose(psi.conj().T @ psi, 1)
    psi = psi.reshape((-1, 1))
    rho = psi @ psi.conj().T
    assert np.isclose(rho.trace(), 1)

    # test density matrix isomorphism
    fermion_op_rho = of.FermionOperator()
    for jj in range(2**m):
        bstring_fop_ket = int_to_ladder_op(jj, m, ket=True)
        for kk in range(2**m):
            bstring_fop_bra = of.hermitian_conjugated(int_to_ladder_op(kk, m, ket=True))
            fermion_op_rho += bstring_fop_ket * bstring_fop_bra * rho[jj, kk]
            test_rho_jk = np.trace(of.get_sparse_operator(bstring_fop_ket * bstring_fop_bra, n_qubits=m).todense().conj().T @ rho)
            # print(np.binary_repr(jj, width=m), np.binary_repr(kk, width=m), 
            #       bstring_fop_ket, bstring_fop_bra)
            assert np.isclose(test_rho_jk, rho[jj, kk])
    rho_op = of.get_sparse_operator(fermion_op_rho).todense()
    assert np.allclose(rho_op, rho)


    # test density matrix isomorphism but with zero indices on outside
    fermion_op_rho = of.FermionOperator()
    for jj in range(2**m):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, m)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**m):
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, m)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            fermion_op_rho += bstring_fop_ket * bstring_fop_bra * zero_fop * rho[jj, kk]
            test_rho_jk = np.trace(of.get_sparse_operator(bstring_fop_ket * bstring_fop_bra * zero_fop, n_qubits=m).todense().conj().T @ rho)
            assert np.isclose(test_rho_jk, rho[jj, kk])

    rho_op = of.get_sparse_operator(fermion_op_rho).todense()
    assert np.allclose(rho_op, rho)

    jj = 6
    kk = 10
    m = 4
    print("|", np.binary_repr(jj, m), "><", np.binary_repr(kk, m), "|")
    bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, m)
    bra_op, bra_zeros = int_to_ladder_op_new_order(kk, m)
    bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
    bstring_fop_bra = of.hermitian_conjugated(bra_op)
    total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
    zero_fop = get_zero_ops(total_zeros)
    A_jk = bstring_fop_ket * bstring_fop_bra * zero_fop
    print("A_jk ", A_jk)

    # test channel isomorphism for particle conserving case
    phi_channel = of.FermionOperator()
    for ii in range(m):
        phi_channel += of.FermionOperator((ii, 0)) * A_jk * of.FermionOperator((ii, 1))
        print("{} A_jk {}^".format(ii, ii))
        print(of.FermionOperator((ii, 0)) * A_jk * of.FermionOperator((ii, 1)))
        print(of.normal_ordered(of.FermionOperator((ii, 0)) * A_jk * of.FermionOperator((ii, 1))))
        print()

    # print("Channel")
    # print(phi_channel)
    # print("Normal Ordered")
    # print(of.normal_ordered(phi_channel))
    c_true = of.FermionOperator("1^ 2 2^ 0 3 3^")
    # print(c_true)
    assert np.allclose(of.get_sparse_operator(c_true, n_qubits=m).todense(), 
                      of.get_sparse_operator(phi_channel, n_qubits=m).todense())


    # particle violating case. We see that the sign isn't always positive
    c1 = of.FermionOperator(((3, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 0), (3, 0), (0, 0), (3, 1)))
    c2 = of.FermionOperator(((1, 1), (2, 1), (3, 0), (4, 1), (5, 1), (6, 1), (7, 0), (3, 1), (0, 0)))
    assert np.allclose(of.get_sparse_operator(c1, n_qubits=8).todense(), -of.get_sparse_operator(c2, n_qubits=8).todense())


    # check marginalization signs for particle conserving case
    print("Check particle conserving case")
    m = 5
    for jj in range(2**m):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, m)
        jj_ket = np.binary_repr(jj, m)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**m):
            kk_ket = np.binary_repr(kk, m)
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, m)
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            A_jk = bstring_fop_ket * bstring_fop_bra * zero_fop

            if sum(map(int, jj_ket)) == sum(map(int, kk_ket)) == 3:
                phi_channel = of.FermionOperator()
                for ii in range(m):
                    ii_string = [0] * m
                    ii_string[ii] = 1
                    ii_string = "".join([str(xx) for xx in ii_string])
                    if jj_ket[ii] == kk_ket[ii] == '1':
                        marginalized_ket, ket_phase = change_raise_lowering(bstring_fop_ket, ii)
                        marginalized_bra, bra_phase = change_raise_lowering(bstring_fop_bra, ii)
                        marginalized_A_jk = marginalized_ket * marginalized_bra * zero_fop
                        print(ket_phase, marginalized_ket, bra_phase, marginalized_bra, zero_fop)
                    else:
                        marginalized_A_jk = 0 * of.FermionOperator(())

                    phi_channel = of.FermionOperator((ii, 0)) * A_jk * of.FermionOperator((ii, 1))
                    # print(phi_channel)
                    # print(marginalized_A_jk)
                    print(np.allclose(of.get_sparse_operator(marginalized_A_jk, n_qubits=m).todense(),
                                       of.get_sparse_operator(phi_channel, n_qubits=m).todense())
                    )

                    assert np.allclose(of.get_sparse_operator(marginalized_A_jk, n_qubits=m).todense(),
                                       of.get_sparse_operator(phi_channel, n_qubits=m).todense())
                # print("------------")

def nparticle_density():
    m = 4
    num_electrons = 3
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T

    non_zero_idx = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)

    fermion_op_rho = of.FermionOperator()
    for jj in range(2**num_qubits):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**num_qubits):
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            fermion_op_rho += bstring_fop_ket * bstring_fop_bra * zero_fop * rho[jj, kk]
            
            # test_rho_jk = np.trace(of.get_sparse_operator(bstring_fop_ket * bstring_fop_bra * zero_fop, n_qubits=num_qubits).todense().conj().T @ rho)
            # assert np.isclose(test_rho_jk, rho[jj, kk])

            if np.binary_repr(jj, width=num_qubits).count('1') == np.binary_repr(kk, width=num_qubits).count('1') == num_electrons:
                test_rho_jk = np.trace(of.get_sparse_operator(bstring_fop_ket * bstring_fop_bra, n_qubits=num_qubits).todense().conj().T @ rho)
                assert np.isclose(test_rho_jk, rho[jj, kk])
                print(bstring_fop_ket * bstring_fop_bra)
                # jj_idx = np.where(np.array([str(xx) for xx in np.binary_repr(jj)][::-1]) == '1')[0]
                # kk_idx =np.where(np.array([str(xx) for xx in np.binary_repr(jj)][::-1]) == '1')[0]
                # daggered_rdm_op = of.hermitian_conjugated(bstring_fop_ket * bstring_fop_bra)
                # rdm_idx = [xx[0] for xx in list(daggered_rdm_op.terms.keys())[0]]
                # assert np.isclose(test_rho_jk, rho[jj, kk])
                # if num_electrons == 2:
                #     assert np.isclose(tpdm[rdm_idx[0], rdm_idx[1], rdm_idx[2], rdm_idx[3]], test_rho_jk)
                #     if jj == kk:
                #         print(rdm_idx, tpdm[rdm_idx[0], rdm_idx[1], rdm_idx[2], rdm_idx[3]], test_rho_jk)


    rho_op = of.get_sparse_operator(fermion_op_rho).todense()
    rho_op = rho_op[:, non_zero_idx]
    rho_op = rho_op[non_zero_idx, :]

    print(rho.trace())
    rho = rho[:, non_zero_idx]
    rho = rho[non_zero_idx, :]

    assert np.allclose(rho_op, rho)
    print(rho.trace(), comb(num_electrons, num_electrons ))
          
def marginalization():
    m = 3
    num_electrons = 2
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T

    non_zero_idx = []
    margin_basis = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)
        if np.binary_repr(jj).count('1') == num_electrons - 1:
            margin_basis.append(jj)

    # get the algebraic representation 
    fermion_op_rho = of.FermionOperator()
    for jj in range(2**num_qubits):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**num_qubits):
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            fermion_op_rho += bstring_fop_ket * bstring_fop_bra * zero_fop * rho[jj, kk]
    assert np.allclose(of.get_sparse_operator(fermion_op_rho).todense(), rho)

    # apply the channel
    phi = of.FermionOperator()
    for ll in range(num_qubits):
        phi += of.FermionOperator((ll, 0)) * fermion_op_rho * of.FermionOperator((ll, 1))
    phi_mat = of.get_sparse_operator(phi, n_qubits=num_qubits).todense()

    phi_mat_margin = phi_mat[:, margin_basis]
    phi_mat_margin = phi_mat_margin[margin_basis, :]

    # Explicitly evaluate the elements of phi(rho) by 
    # evaluating the summation 
    for jj in range(2**m):
        jj_ket = np.binary_repr(jj, width=num_qubits)
        jj_ket_ints = list([int(xx) for xx in jj_ket])
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**m):
            kk_ket = np.binary_repr(kk, width=num_qubits)
            kk_ket_ints = list([int(xx) for xx in kk_ket])
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            if jj_ket.count('1') == kk_ket.count('1') == num_electrons - 1:
                # put the ll term in the string and calculate density matrix element
                test_val = 0
                for ll in range(num_qubits):
                    if jj_ket_ints[ll] == 0 and kk_ket_ints[ll] == 0:
                        lifted_jj_ket_ints = list(jj_ket_ints)
                        lifted_kk_ket_ints = list(kk_ket_ints)
                        lifted_jj_ket_ints[ll] = 1
                        lifted_jj_phase = (-1)**sum(lifted_jj_ket_ints[:ll])
                        lifted_kk_ket_ints[ll] = 1
                        lifted_kk_phase = (-1)**sum(lifted_kk_ket_ints[:ll])
                        # now convert the ket-ints back into the expectation value * the phases
                        lifted_jj_ket = of.FermionOperator(([(int(xx), 1) for xx in np.argwhere(lifted_jj_ket_ints).ravel()]))
                        lifted_kk_ket = of.FermionOperator(([(int(xx), 1) for xx in np.argwhere(lifted_kk_ket_ints).ravel()]))
                        print(lifted_jj_phase, lifted_jj_ket_ints, np.argwhere(lifted_jj_ket_ints).ravel(), lifted_jj_ket,
                              lifted_kk_phase, lifted_kk_ket_ints, np.argwhere(lifted_kk_ket_ints).ravel(), lifted_kk_ket)
                        lifted_op = lifted_jj_ket * of.hermitian_conjugated(lifted_kk_ket)
                        test_val += lifted_jj_phase * lifted_kk_phase * np.trace(of.get_sparse_operator(lifted_op, n_qubits=num_qubits).todense().conj().T @ rho)

                assert np.isclose(test_val, phi_mat[jj, kk])

                # second version that is more commensurate with how
                # we usually write contractions
                test_val2 = 0
                for ll in range(num_qubits):
                    fop_to_sum = of.FermionOperator(((ll, 1))) * bstring_fop_ket * bstring_fop_bra * of.FermionOperator(((ll, 0)))
                    tmp_fop = of.get_sparse_operator(fop_to_sum, n_qubits=num_qubits).todense().conj().T
                    test_val2 += np.trace(tmp_fop @ rho)
                print(test_val2, phi_mat[jj, kk])
                assert np.isclose(test_val2, phi_mat[jj, kk])


def trace_condition():
    m = 4
    num_electrons = 3
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T

    non_zero_idx = []
    margin_basis = []
    double_margin_basis = []
    for jj in range(2**num_qubits):
        if np.binary_repr(jj).count('1') == num_electrons:
            non_zero_idx.append(jj)
        if np.binary_repr(jj).count('1') == num_electrons - 1:
            margin_basis.append(jj)
        if np.binary_repr(jj).count('1') == num_electrons - 2:
            double_margin_basis.append(jj)

    # get the algebraic representation 
    fermion_op_rho = of.FermionOperator()
    for jj in range(2**num_qubits):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        for kk in range(2**num_qubits):
            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)
            fermion_op_rho += bstring_fop_ket * bstring_fop_bra * zero_fop * rho[jj, kk]
    assert np.allclose(of.get_sparse_operator(fermion_op_rho).todense(), rho)

    # apply the channel
    phi = of.FermionOperator()
    for ll in range(num_qubits):
        phi += of.FermionOperator((ll, 0)) * fermion_op_rho * of.FermionOperator((ll, 1)) * (1/num_electrons)
    phi_mat = of.get_sparse_operator(phi, n_qubits=num_qubits).todense()

    phi_mat_margin = phi_mat[:, margin_basis]
    phi_mat_margin = phi_mat_margin[margin_basis, :]

    print(phi_mat_margin.trace())


    # apply the channel
    phi2 = of.FermionOperator()
    for ll in range(num_qubits):
        phi2 += of.FermionOperator((ll, 0)) * phi * of.FermionOperator((ll, 1)) * (1 / (num_electrons - 1) )
    phi2_mat = of.get_sparse_operator(phi2, n_qubits=num_qubits).todense()
    phi2_mat_margin = phi2_mat[:, double_margin_basis]
    phi2_mat_margin = phi2_mat_margin[double_margin_basis, :]
    print(phi2_mat_margin.trace())

def is_zero(of_op):
    """
    Check if OpenFermion Operator is zero 
    """ 
    if len(of_op.terms) == 0:
        return True
    elif len(of_op.terms) == 1:
        return np.isclose(list(of_op.terms.items())[0][1], 0)
    else:
        return False

def majorana_representation():
    m = 3
    num_electrons = 2
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T

    # test density matrix isomorphism but with zero indices on outside
    fermion_op_rho = of.FermionOperator()
    total_count = 0
    for jj in range(2**num_qubits):
        bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
        ket_zero_fop = get_zero_ops(ket_zeros)
        jj_ket = np.binary_repr(jj, width=num_qubits)
        jja_ket = jj_ket[::2]
        jjb_ket = jj_ket[1::2]
        sz_jj_ket = jja_ket.count('1') - jjb_ket.count('1')
        for kk in range(jj, 2**num_qubits):
            kk_ket = np.binary_repr(kk, width=num_qubits)
            kka_ket = kk_ket[::2]
            kkb_ket = kk_ket[1::2]
            sz_kk_ket = kka_ket.count('1') - kkb_ket.count('1')

            bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
            bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
            bstring_fop_bra = of.hermitian_conjugated(bra_op)
            total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
            zero_fop = get_zero_ops(total_zeros)

            AIJ_P_AJI = 0.5 * (bstring_fop_ket * bstring_fop_bra * zero_fop + bra_op * of.hermitian_conjugated(bstring_fop_ket) * zero_fop)
            AIJ_M_AJI = 0.5j * (bstring_fop_ket * bstring_fop_bra * zero_fop  - bra_op * of.hermitian_conjugated(bstring_fop_ket) * zero_fop)
            assert of.is_hermitian(AIJ_P_AJI)
            assert of.is_hermitian(AIJ_M_AJI)
            test_rho_jk_real = np.trace(of.get_sparse_operator(AIJ_P_AJI, n_qubits=num_qubits) @ rho)
            if not np.isclose(test_rho_jk_real, 0) or not np.isclose(rho[jj, kk].real, 0):
                assert np.isclose(test_rho_jk_real, rho[jj, kk].real)
            test_rho_jk_imag = np.trace(of.get_sparse_operator(AIJ_M_AJI, n_qubits=num_qubits) @ rho)
            if not np.isclose(test_rho_jk_imag, 0) or not np.isclose(rho[jj, kk].imag, 0):
                assert np.isclose(test_rho_jk_imag, rho[jj, kk].imag)
                if jj == kk:
                    assert np.isclose(test_rho_jk_imag, 0)

            if jj != kk: 
                fermion_op_rho += AIJ_P_AJI * rho[jj, kk].real * 2
                fermion_op_rho += AIJ_M_AJI * rho[jj, kk].imag * 2
            else:
                fermion_op_rho += AIJ_P_AJI * rho[jj, kk].real
            if np.binary_repr(jj).count('1') == num_electrons and np.binary_repr(kk).count('1') == num_electrons and sz_jj_ket == sz_kk_ket == sz:
                AIJ_P_AJI = 0.5 * (bstring_fop_ket * bstring_fop_bra + bra_op * of.hermitian_conjugated(bstring_fop_ket))
                AIJ_M_AJI = 0.5j * (bstring_fop_ket * bstring_fop_bra  - bra_op * of.hermitian_conjugated(bstring_fop_ket))
                print("Positive")
                print(AIJ_P_AJI)
                print(of.get_majorana_operator(AIJ_P_AJI))
                print("Negative")
                print(AIJ_M_AJI)
                print(of.get_majorana_operator(AIJ_M_AJI))
                print()
                total_count += 1

    rho_op = of.get_sparse_operator(fermion_op_rho).todense()
    assert np.allclose(rho_op, rho)
    print(total_count)

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



def identity_2_representation():
    m = 3
    num_electrons = 2
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


            if jj_ket.count('1') == 2 and kk_ket.count('1') == 2 and jj_ket == kk_ket:
                # print(bstring_fop_ket * bstring_fop_bra)
                fermion_op_identity += bstring_fop_ket * bstring_fop_bra * zero_fop
    
    identity_mat = of.get_sparse_operator(fermion_op_identity).todense()

    compliment_mat = identity_mat.copy()
    compliment_mat = compliment_mat[:, compliment_idx]
    compliment_mat = compliment_mat[compliment_idx, :]
    assert np.allclose(compliment_mat, 0)

    identity_mat = identity_mat[:, non_zero_idx]
    identity_mat = identity_mat[non_zero_idx, :]
    assert np.allclose(identity_mat.real, np.eye(int(comb(int(num_qubits), 2))))

    
    # Now construct the operator from the wedge product form
    # |i ^ k><j ^ l| delta^{i}_{j}delta^{k}_{l} / 4
    test_ident_fop = of.FermionOperator()
    for i, j, k, l in itertools.product(range(num_qubits), repeat=4):
        if i == k or j == l:
            continue
        ket_fop = of.FermionOperator(([(xx, 1) for xx in [i, k]]))
        bra_fop = of.hermitian_conjugated(of.FermionOperator(([(xx, 1) for xx in [j, l]])))
        zero_set = sorted(list(set([i, j, k, l]) ^ set(range(num_qubits))))
        zero_fop = get_zero_ops(zero_set)
        # print(ket_fop, bra_fop, zero_fop)
        # (0, 1), (0, 1)^ delta(0, 0) delta(1, 1)
        # (0, 1), (1, 0)^ delta(0, 1) delta(1, 0)
        # (1, 0), (0, 1)^ delta(1, 0) delta(0, 1)
        # (1, 0), (1, 0)^ delta(1, 1) delta(0, 0)
        test_ident_fop += ket_fop * bra_fop * zero_fop * kdelta(i, j) * kdelta(k, l) / 2 # hmm...not 4?

    print(test_ident_fop)

    identity_mat = of.get_sparse_operator(test_ident_fop).todense()

    compliment_mat = identity_mat.copy()
    compliment_mat = compliment_mat[:, compliment_idx]
    compliment_mat = compliment_mat[compliment_idx, :]
    assert np.allclose(compliment_mat, 0)

    identity_mat = identity_mat[:, non_zero_idx]
    identity_mat = identity_mat[non_zero_idx, :]
    print(identity_mat.real)
    assert np.allclose(identity_mat.real, np.eye(int(comb(int(num_qubits), 2))))

def identity_3_representation():
    m = 3
    num_electrons = 3
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


            if jj_ket.count('1') == num_electrons and kk_ket.count('1') == num_electrons and jj_ket == kk_ket:
                # print(bstring_fop_ket * bstring_fop_bra)
                fermion_op_identity += bstring_fop_ket * bstring_fop_bra * zero_fop
    
    identity_mat = of.get_sparse_operator(fermion_op_identity).todense()

    compliment_mat = identity_mat.copy()
    compliment_mat = compliment_mat[:, compliment_idx]
    compliment_mat = compliment_mat[compliment_idx, :]
    assert np.allclose(compliment_mat, 0)

    identity_mat = identity_mat[:, non_zero_idx]
    identity_mat = identity_mat[non_zero_idx, :]
    assert np.allclose(identity_mat.real, np.eye(int(comb(int(num_qubits), num_electrons))))

    
    # Now construct the operator from the wedge product form
    # delta^{i}_{j} delta^{k}_{l} delta^{m}_{n}
    # |i ^ k><j ^ l| delta^{i}_{j}delta^{k}_{l} / 4
    test_ident_fop = of.FermionOperator()
    for i, j, k, l, m, n in itertools.product(range(num_qubits), repeat=6):
        ket_fop = of.FermionOperator(([(xx, 1) for xx in [i, k, m]]))
        bra_fop = of.hermitian_conjugated(of.FermionOperator(([(xx, 1) for xx in [j, l, n]])))
        zero_set = sorted(list(set([i, j, k, l, m, n]) ^ set(range(num_qubits))))
        zero_fop = get_zero_ops(zero_set)
        test_ident_fop += ket_fop * bra_fop * zero_fop * kdelta(i, j) * kdelta(k, l) * kdelta(m, n) / factorial(3) # hmm...not 4?

    identity_mat = of.get_sparse_operator(test_ident_fop).todense()

    compliment_mat = identity_mat.copy()
    compliment_mat = compliment_mat[:, compliment_idx]
    compliment_mat = compliment_mat[compliment_idx, :]
    assert np.allclose(compliment_mat, 0)

    identity_mat = identity_mat[:, non_zero_idx]
    identity_mat = identity_mat[non_zero_idx, :]
    print(identity_mat.real.shape, comb(int(num_qubits), num_electrons))
    assert np.allclose(identity_mat.real, np.eye(int(comb(int(num_qubits), num_electrons))))

def lifted_one_body_operator():
    np.set_printoptions(linewidth=500)
    np.random.seed(10)
    m = 3
    num_electrons = 2
    sz = 0 if num_electrons % 2 == 0 else 1
    num_qubits = 2 * m
    fqe_wf = fqe.Wavefunction([[num_electrons, sz, m], [num_electrons, -sz, m]])
    fqe_wf.set_wfn(strategy='random')
    fqe_wf.print_wfn()
    fqe_data = fqe_wf.sector((num_electrons, sz))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    wf = fqe.to_cirq(fqe_wf).reshape((-1, 1))
    rho = wf @ wf.conj().T

    A = np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits)) \
         + 1j * np.random.randn(num_qubits**2).reshape((num_qubits, num_qubits))
    A = A + A.conj().T
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
    A_fop_mat = A_fop_mat[:, non_zero_idx]
    A_fop_mat = A_fop_mat[non_zero_idx, :]


    # fermion_op = of.FermionOperator()
    # test_A_mat = np.zeros((2**num_qubits, 2**num_qubits))
    # for jj in range(2**num_qubits):
    #     bstring_fop_ket, ket_zeros = int_to_ladder_op_new_order(jj, num_qubits)
    #     ket_zero_fop = get_zero_ops(ket_zeros)
    #     jj_ket = np.binary_repr(jj, width=num_qubits)
    #     jja_ket = jj_ket[::2]
    #     jjb_ket = jj_ket[1::2]
    #     sz_jj_ket = jja_ket.count('1') - jjb_ket.count('1')

    #     for kk in range(2**num_qubits):
    #         bra_op, bra_zeros = int_to_ladder_op_new_order(kk, num_qubits)
    #         bra_zero_fop = of.hermitian_conjugated(get_zero_ops(bra_zeros))
    #         bstring_fop_bra = of.hermitian_conjugated(bra_op)
    #         total_zeros = sorted(list(set(ket_zeros) & set(bra_zeros))) # the & symbol is set intersection
    #         zero_fop = get_zero_ops(total_zeros)
    #         kk_ket = np.binary_repr(kk, width=num_qubits)
    #         kka_ket = kk_ket[::2]
    #         kkb_ket = kk_ket[1::2]
    #         sz_kk_ket = kka_ket.count('1') - kkb_ket.count('1')

    #         if jj_ket.count('1') == num_electrons and kk_ket.count('1') == num_electrons:
    #             # # get the jj_ket indices and kk_ket indices
    #             jj_ket_idx = np.argwhere(list(map(int, jj_ket))).ravel()
    #             i, k = jj_ket_idx
    #             kk_ket_idx = np.argwhere(list(map(int, kk_ket))).ravel()
    #             j, l = kk_ket_idx
    #             fermion_op += bstring_fop_ket * bstring_fop_bra * zero_fop * A[k, l] * kdelta(i, j) / 4
    #             # print(bstring_fop_ket, bstring_fop_bra)
    #             # test_A_mat[jj, kk] = A[i, j] * kdelta(k, l) / 4
    
    # fermion_op_mat = of.get_sparse_operator(fermion_op).todense()

    # fermion_op_mat = fermion_op_mat[:, non_zero_idx]
    # fermion_op_mat = fermion_op_mat[non_zero_idx, :]

    # test_A_mat = test_A_mat[:, non_zero_idx]
    # test_A_mat = test_A_mat[non_zero_idx, :]
    # # print(fermion_op_mat.real)
    # # print(A_fop_mat)
    # # print(test_A_mat)
    # assert np.allclose(test_A_mat, fermion_op_mat)
   


    A_op_pre_wedge = of.FermionOperator()
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


            if jj_ket.count('1') == 1 and kk_ket.count('1') == 1:
                jj_ket_idx = np.argwhere(list(map(int, jj_ket))).ravel()
                i = jj_ket_idx[0]
                kk_ket_idx = np.argwhere(list(map(int, kk_ket))).ravel()
                j = kk_ket_idx[0]

                A_op_pre_wedge += bstring_fop_ket * bstring_fop_bra * zero_fop * A[i, j]

    A_op_pre_wedge_mat = of.get_sparse_operator(A_op_pre_wedge).todense()
    A_op_pre_wedge_mat = A_op_pre_wedge_mat[:, one_idx]
    A_op_pre_wedge_mat = A_op_pre_wedge_mat[one_idx, :]

    A_fop_mat = of.get_sparse_operator(A_fop).todense()
    true_A_one = A_fop_mat.copy()
    true_A_one = true_A_one[:, one_idx]
    true_A_one = true_A_one[one_idx, :]
    print(A_op_pre_wedge_mat)
    print(true_A_one)
    print(np.allclose(A_op_pre_wedge_mat, true_A_one))

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
                fermion_op_identity += bstring_fop_ket * bstring_fop_bra * zero_fop
    
    identity_mat = of.get_sparse_operator(fermion_op_identity).todense()
    
    lifted_A = of.get_sparse_operator(A_op_pre_wedge * fermion_op_identity).todense()
    A_lifted = of.get_sparse_operator(fermion_op_identity * A_op_pre_wedge).todense()

    assert np.allclose(lifted_A, A_lifted)

    print(of.normal_ordered(A_op_pre_wedge * fermion_op_identity))
    exit()

    lifted_A = lifted_A[:, two_idx]
    lifted_A = lifted_A[two_idx, :]

    A_fop_mat = of.get_sparse_operator(A_fop).todense()
    true_A_two = A_fop_mat.copy()
    true_A_two = true_A_two[:, two_idx]
    true_A_two = true_A_two[two_idx, :]

    print(true_A_two)
    print()


   
if __name__ == "__main__":
    # main()
    # nparticle_density()
    # marginalization()
    # trace_condition()
    # majorana_representation()
    identity_representation()
    # identity_2_representation()
    # identity_3_representation()
    # lifted_one_body_operator()


    # dim = 4
    # A = np.random.randn(dim**2).reshape((dim, dim))
    # B = np.random.randn(dim**2).reshape((dim, dim))
    # AB = np.einsum("ij,kl->ikjl", A, B)
    # for i, j, k, l in itertools.product(range(dim), repeat=4):
    #     assert np.isclose(AB[i, k, j, l], A[i, j] * B[k, l])
    #     if (i, j, k, l) == (0, 1, 2, 3):
    #         print(A[i, j] * B[k, l], AB[i, k, j, l])
    #         print(A[k, j] * B[i, l], AB[k, i, j, l])

