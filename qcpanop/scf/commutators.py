import numpy as np


def k2_rotgen_grad_nosymm(k2, p, q, tpdm):
    """
    No symmetry is considered on k2 or the tpdm.

    This is for testing the spin-summed complex version
    :param k2:
    :param p:
    :param q:
    :param tpdm:
    :return:
    """
    expectation = 0. + 1j * 0
    #  (  -1.00000) k2(p,a,b,c) cre(q) cre(a) des(b) des(c)
    expectation += -1.0 * np.einsum('abc,abc', k2[p, :, :, :], tpdm[q, :, :, :])
    #  (   1.00000) k2(q,a,b,c) cre(p) cre(a) des(b) des(c)
    expectation += 1.0 * np.einsum('abc,abc', k2[q, :, :, :], tpdm[p, :, :, :])
    #  (   1.00000) k2(a,p,b,c) cre(q) cre(a) des(b) des(c)
    expectation += 1.0 * np.einsum('abc,abc', k2[:, p, :, :], tpdm[q, :, :, :])
    #  (  -1.00000) k2(a,q,b,c) cre(p) cre(a) des(b) des(c)
    expectation += -1.0 * np.einsum('abc,abc', k2[:, q, :, :], tpdm[p, :, :, :])
    #  (  -1.00000) k2(a,b,p,c) cre(a) cre(b) des(q) des(c)
    expectation += -1.0 * np.einsum('abc,abc', k2[:, :, p, :], tpdm[:, :, q, :])
    #  (   1.00000) k2(a,b,q,c) cre(a) cre(b) des(p) des(c)
    expectation += 1.0 * np.einsum('abc,abc', k2[:, :, q, :], tpdm[:, :, p, :])
    #  (   1.00000) k2(a,b,c,p) cre(a) cre(b) des(q) des(c)
    expectation += 1.0 * np.einsum('abc,abc', k2[:, :, :, p], tpdm[:, :, q, :])
    #  (  -1.00000) k2(a,b,c,q) cre(a) cre(b) des(p) des(c)
    expectation += -1.0 * np.einsum('abc,abc', k2[:, :, :, q], tpdm[:, :, p, :])
    return expectation


def k2_rotgen_hess_nosymm(k2, p, q, r, s, tpdm):
    """
    No symmetry is considered in k2 or tpdm
    """
    expectation = 0. + 1j * 0.
    #  (  -1.00000) k2(p,r,a,b) cre(q) cre(s) des(a) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[q, s, :, :])
    #  (   1.00000) k2(p,s,a,b) cre(q) cre(r) des(a) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[q, r, :, :])
    #  (  -1.00000) k2(p,a,r,b) cre(q) cre(a) des(s) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[p, :, r, :], tpdm[q, :, s, :])
    #  (   1.00000) k2(p,a,s,b) cre(q) cre(a) des(r) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[p, :, s, :], tpdm[q, :, r, :])
    #  (   1.00000) k2(p,a,b,r) cre(q) cre(a) des(s) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[p, :, :, r], tpdm[q, :, s, :])
    #  (  -1.00000) k2(p,a,b,s) cre(q) cre(a) des(r) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[p, :, :, s], tpdm[q, :, r, :])
    #  (   1.00000) k2(q,r,a,b) cre(p) cre(s) des(a) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[p, s, :, :])
    #  (  -1.00000) k2(q,s,a,b) cre(p) cre(r) des(a) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[p, r, :, :])
    #  (   1.00000) k2(q,a,r,b) cre(p) cre(a) des(s) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[q, :, r, :], tpdm[p, :, s, :])
    #  (  -1.00000) k2(q,a,s,b) cre(p) cre(a) des(r) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[q, :, s, :], tpdm[p, :, r, :])
    #  (  -1.00000) k2(q,a,b,r) cre(p) cre(a) des(s) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[q, :, :, r], tpdm[p, :, s, :])
    #  (   1.00000) k2(q,a,b,s) cre(p) cre(a) des(r) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[q, :, :, s], tpdm[p, :, r, :])
    #  (   1.00000) k2(r,p,a,b) cre(q) cre(s) des(a) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[r, p, :, :], tpdm[q, s, :, :])
    #  (  -1.00000) k2(r,q,a,b) cre(p) cre(s) des(a) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[r, q, :, :], tpdm[p, s, :, :])
    #  (  -1.00000) k2(r,a,p,b) cre(s) cre(a) des(q) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[r, :, p, :], tpdm[s, :, q, :])
    #  (   1.00000) k2(r,a,q,b) cre(s) cre(a) des(p) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[r, :, q, :], tpdm[s, :, p, :])
    #  (   1.00000) k2(r,a,b,p) cre(s) cre(a) des(q) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[r, :, :, p], tpdm[s, :, q, :])
    #  (  -1.00000) k2(r,a,b,q) cre(s) cre(a) des(p) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[r, :, :, q], tpdm[s, :, p, :])
    #  (  -1.00000) k2(s,p,a,b) cre(q) cre(r) des(a) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[s, p, :, :], tpdm[q, r, :, :])
    #  (   1.00000) k2(s,q,a,b) cre(p) cre(r) des(a) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[s, q, :, :], tpdm[p, r, :, :])
    #  (   1.00000) k2(s,a,p,b) cre(r) cre(a) des(q) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[s, :, p, :], tpdm[r, :, q, :])
    #  (  -1.00000) k2(s,a,q,b) cre(r) cre(a) des(p) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[s, :, q, :], tpdm[r, :, p, :])
    #  (  -1.00000) k2(s,a,b,p) cre(r) cre(a) des(q) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[s, :, :, p], tpdm[r, :, q, :])
    #  (   1.00000) k2(s,a,b,q) cre(r) cre(a) des(p) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[s, :, :, q], tpdm[r, :, p, :])
    #  (   1.00000) k2(a,p,r,b) cre(q) cre(a) des(s) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, p, r, :], tpdm[q, :, s, :])
    #  (  -1.00000) k2(a,p,s,b) cre(q) cre(a) des(r) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, p, s, :], tpdm[q, :, r, :])
    #  (  -1.00000) k2(a,p,b,r) cre(q) cre(a) des(s) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, p, :, r], tpdm[q, :, s, :])
    #  (   1.00000) k2(a,p,b,s) cre(q) cre(a) des(r) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, p, :, s], tpdm[q, :, r, :])
    #  (  -1.00000) k2(a,q,r,b) cre(p) cre(a) des(s) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, q, r, :], tpdm[p, :, s, :])
    #  (   1.00000) k2(a,q,s,b) cre(p) cre(a) des(r) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, q, s, :], tpdm[p, :, r, :])
    #  (   1.00000) k2(a,q,b,r) cre(p) cre(a) des(s) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, q, :, r], tpdm[p, :, s, :])
    #  (  -1.00000) k2(a,q,b,s) cre(p) cre(a) des(r) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, q, :, s], tpdm[p, :, r, :])
    #  (   1.00000) k2(a,r,p,b) cre(s) cre(a) des(q) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, r, p, :], tpdm[s, :, q, :])
    #  (  -1.00000) k2(a,r,q,b) cre(s) cre(a) des(p) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, r, q, :], tpdm[s, :, p, :])
    #  (  -1.00000) k2(a,r,b,p) cre(s) cre(a) des(q) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, r, :, p], tpdm[s, :, q, :])
    #  (   1.00000) k2(a,r,b,q) cre(s) cre(a) des(p) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, r, :, q], tpdm[s, :, p, :])
    #  (  -1.00000) k2(a,s,p,b) cre(r) cre(a) des(q) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, s, p, :], tpdm[r, :, q, :])
    #  (   1.00000) k2(a,s,q,b) cre(r) cre(a) des(p) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, s, q, :], tpdm[r, :, p, :])
    #  (   1.00000) k2(a,s,b,p) cre(r) cre(a) des(q) des(b)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, s, :, p], tpdm[r, :, q, :])
    #  (  -1.00000) k2(a,s,b,q) cre(r) cre(a) des(p) des(b)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, s, :, q], tpdm[r, :, p, :])
    #  (  -1.00000) k2(a,b,p,r) cre(a) cre(b) des(q) des(s)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, :, p, r], tpdm[:, :, q, s])
    #  (   1.00000) k2(a,b,p,s) cre(a) cre(b) des(q) des(r)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, :, p, s], tpdm[:, :, q, r])
    #  (   1.00000) k2(a,b,q,r) cre(a) cre(b) des(p) des(s)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, :, q, r], tpdm[:, :, p, s])
    #  (  -1.00000) k2(a,b,q,s) cre(a) cre(b) des(p) des(r)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, :, q, s], tpdm[:, :, p, r])
    #  (   1.00000) k2(a,b,r,p) cre(a) cre(b) des(q) des(s)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, :, r, p], tpdm[:, :, q, s])
    #  (  -1.00000) k2(a,b,r,q) cre(a) cre(b) des(p) des(s)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, :, r, q], tpdm[:, :, p, s])
    #  (  -1.00000) k2(a,b,s,p) cre(a) cre(b) des(q) des(r)
    expectation += -1.0 * np.einsum('ab,ab', k2[:, :, s, p], tpdm[:, :, q, r])
    #  (   1.00000) k2(a,b,s,q) cre(a) cre(b) des(p) des(r)
    expectation += 1.0 * np.einsum('ab,ab', k2[:, :, s, q], tpdm[:, :, p, r])
    #  (  -1.00000) k2(p,a,b,c) kdelta(q,r) cre(s) cre(a) des(b) des(c)
    if q == r:
        expectation += -1.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                        tpdm[s, :, :, :])
    #  (   1.00000) k2(p,a,b,c) kdelta(q,s) cre(r) cre(a) des(b) des(c)
    if q == s:
        expectation += 1.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                       tpdm[r, :, :, :])
    #  (   1.00000) k2(q,a,b,c) kdelta(p,r) cre(s) cre(a) des(b) des(c)
    if p == r:
        expectation += 1.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                       tpdm[s, :, :, :])
    #  (  -1.00000) k2(q,a,b,c) kdelta(p,s) cre(r) cre(a) des(b) des(c)
    if p == s:
        expectation += -1.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                        tpdm[r, :, :, :])
    #  (   1.00000) k2(a,p,b,c) kdelta(q,r) cre(s) cre(a) des(b) des(c)
    if q == r:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, p, :, :],
                                       tpdm[s, :, :, :])
    #  (  -1.00000) k2(a,p,b,c) kdelta(q,s) cre(r) cre(a) des(b) des(c)
    if q == s:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, p, :, :],
                                        tpdm[r, :, :, :])
    #  (  -1.00000) k2(a,q,b,c) kdelta(p,r) cre(s) cre(a) des(b) des(c)
    if p == r:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, q, :, :],
                                        tpdm[s, :, :, :])
    #  (   1.00000) k2(a,q,b,c) kdelta(p,s) cre(r) cre(a) des(b) des(c)
    if p == s:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, q, :, :],
                                       tpdm[r, :, :, :])
    #  (  -1.00000) k2(a,b,p,c) kdelta(q,r) cre(a) cre(b) des(s) des(c)
    if q == r:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, :, p, :],
                                        tpdm[:, :, s, :])
    #  (   1.00000) k2(a,b,p,c) kdelta(q,s) cre(a) cre(b) des(r) des(c)
    if q == s:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, :, p, :],
                                       tpdm[:, :, r, :])
    #  (   1.00000) k2(a,b,q,c) kdelta(p,r) cre(a) cre(b) des(s) des(c)
    if p == r:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, :, q, :],
                                       tpdm[:, :, s, :])
    #  (  -1.00000) k2(a,b,q,c) kdelta(p,s) cre(a) cre(b) des(r) des(c)
    if p == s:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, :, q, :],
                                        tpdm[:, :, r, :])
    #  (   1.00000) k2(a,b,c,p) kdelta(q,r) cre(a) cre(b) des(s) des(c)
    if q == r:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, :, :, p],
                                       tpdm[:, :, s, :])
    #  (  -1.00000) k2(a,b,c,p) kdelta(q,s) cre(a) cre(b) des(r) des(c)
    if q == s:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, :, :, p],
                                        tpdm[:, :, r, :])
    #  (  -1.00000) k2(a,b,c,q) kdelta(p,r) cre(a) cre(b) des(s) des(c)
    if p == r:
        expectation += -1.0 * np.einsum('abc,abc', k2[:, :, :, q],
                                        tpdm[:, :, s, :])
    #  (   1.00000) k2(a,b,c,q) kdelta(p,s) cre(a) cre(b) des(r) des(c)
    if p == s:
        expectation += 1.0 * np.einsum('abc,abc', k2[:, :, :, q],
                                       tpdm[:, :, r, :])
    return expectation

def k2_rotgen_grad_one_body(h1, p, q, opdm):
    """
    spin orbital representation

    <psi|[h_{pq}, pq]|psi>

    :param h1: spin-orbital 1-electron integrals
    :param p: spin-orbital index for p^q - q^p operator
    :param q: spin-orbital index for p^q - q^p operator
    :param opdm: spin-orbital opdm ordered alpha,beta,alpha,beta....
                 OpenFermion ordering.
    """
    expectation = 0.
    #  (   1.00000) h1(p,a) cre(q) des(a)
    expectation += 1.0 * np.einsum('a,a', h1[p, :], opdm[q, :])
    #  (   1.00000) h1(a,p) cre(a) des(q)
    expectation += 1.0 * np.einsum('a,a', h1[:, p], opdm[:, q])
    #  (  -1.00000) h1(q,a) cre(p) des(a)
    expectation += -1.0 * np.einsum('a,a', h1[q, :], opdm[p, :])
    #  (  -1.00000) h1(a,q) cre(a) des(p)
    expectation += -1.0 * np.einsum('a,a', h1[:, q], opdm[:, p])
    return expectation


def k2_rotgen_grad(k2, p, q, tpdm):
    """
    k2-tensor such such that the following terms correspond

    k2[p, q, s, r] p^ q^ r s
    sum_{pqrs}k2_{pqsr}[p^q^ r s, m^ n]

    This can probably sped up with blas call to vector dot on reshaped and
    flattened k2 and tpdm

    :param k2: reduced Hamiltonian in OpenFermion ordering. k2 shoudl be
               antisymmetric in pq and rs indices.
    :param p: spin-orbital index for p^q - q^p operator
    :param q: spin-orbital index for p^q - q^p operator
    :param tpdm: spin-orbital tpdm in OpenFermion ordering
    """
    expectation = 0.
    #  (  -2.00000) k2(p,a,b,c) cre(q) cre(a) des(b) des(c)
    expectation += -2.0 * np.einsum('abc,abc', k2[p, :, :, :], tpdm[q, :, :, :])
    #  (  -2.00000) k2(p,a,b,c) cre(b) cre(c) des(q) des(a)
    expectation += -2.0 * np.einsum('abc,bca', k2[p, :, :, :], tpdm[:, :, q, :])
    #  (   2.00000) k2(q,a,b,c) cre(p) cre(a) des(b) des(c)
    expectation += 2.0 * np.einsum('abc,abc', k2[q, :, :, :], tpdm[p, :, :, :])
    #  (   2.00000) k2(q,a,b,c) cre(b) cre(c) des(p) des(a)
    expectation += 2.0 * np.einsum('abc,bca', k2[q, :, :, :], tpdm[:, :, p, :])
    return expectation


def k2_rotgen_hessian(k2, p, q, r, s, tpdm):
    """
    <| [[k2, p^ q - q^ p], r^s - s^r]|>

    This can be sped up with a dgemm call through numpy

    :param k2: reduced Hamiltonian in OpenFermion ordering. k2 shoudl be
               antisymmetric in pq and rs indices.
    :param p: spin-orbital index for p^q - q^p operator
    :param q: spin-orbital index for p^q - q^p operator
    :param p: spin-orbital index for r^s - s^r operator
    :param q: spin-orbital index for r^s - s^r operator
    :param tpdm: spin-orbital tpdm in OpenFermion ordering
    """
    expectation = 0.
    #  (  -2.00000) k2(p,r,a,b) cre(q) cre(s) des(a) des(b)
    expectation += -2.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[q, s, :, :])
    #  (  -2.00000) k2(p,r,a,b) cre(a) cre(b) des(q) des(s)
    expectation += -2.0 * np.einsum('ab,ab', k2[p, r, :, :], tpdm[:, :, q, s])
    #  (   2.00000) k2(p,s,a,b) cre(q) cre(r) des(a) des(b)
    expectation += 2.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[q, r, :, :])
    #  (   2.00000) k2(p,s,a,b) cre(a) cre(b) des(q) des(r)
    expectation += 2.0 * np.einsum('ab,ab', k2[p, s, :, :], tpdm[:, :, q, r])
    #  (  -4.00000) k2(p,a,r,b) cre(q) cre(a) des(s) des(b)
    expectation += -4.0 * np.einsum('ab,ab', k2[p, :, r, :], tpdm[q, :, s, :])
    #  (  -4.00000) k2(p,a,r,b) cre(s) cre(b) des(q) des(a)
    expectation += -4.0 * np.einsum('ab,ba', k2[p, :, r, :], tpdm[s, :, q, :])
    #  (   4.00000) k2(p,a,s,b) cre(q) cre(a) des(r) des(b)
    expectation += 4.0 * np.einsum('ab,ab', k2[p, :, s, :], tpdm[q, :, r, :])
    #  (   4.00000) k2(p,a,s,b) cre(r) cre(b) des(q) des(a)
    expectation += 4.0 * np.einsum('ab,ba', k2[p, :, s, :], tpdm[r, :, q, :])
    #  (   2.00000) k2(q,r,a,b) cre(p) cre(s) des(a) des(b)
    expectation += 2.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[p, s, :, :])
    #  (   2.00000) k2(q,r,a,b) cre(a) cre(b) des(p) des(s)
    expectation += 2.0 * np.einsum('ab,ab', k2[q, r, :, :], tpdm[:, :, p, s])
    #  (  -2.00000) k2(q,s,a,b) cre(p) cre(r) des(a) des(b)
    expectation += -2.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[p, r, :, :])
    #  (  -2.00000) k2(q,s,a,b) cre(a) cre(b) des(p) des(r)
    expectation += -2.0 * np.einsum('ab,ab', k2[q, s, :, :], tpdm[:, :, p, r])
    #  (   4.00000) k2(q,a,r,b) cre(p) cre(a) des(s) des(b)
    expectation += 4.0 * np.einsum('ab,ab', k2[q, :, r, :], tpdm[p, :, s, :])
    #  (   4.00000) k2(q,a,r,b) cre(s) cre(b) des(p) des(a)
    expectation += 4.0 * np.einsum('ab,ba', k2[q, :, r, :], tpdm[s, :, p, :])
    #  (  -4.00000) k2(q,a,s,b) cre(p) cre(a) des(r) des(b)
    expectation += -4.0 * np.einsum('ab,ab', k2[q, :, s, :], tpdm[p, :, r, :])
    #  (  -4.00000) k2(q,a,s,b) cre(r) cre(b) des(p) des(a)
    expectation += -4.0 * np.einsum('ab,ba', k2[q, :, s, :], tpdm[r, :, p, :])
    #  (  -2.00000) k2(p,a,b,c) kdelta(q,r) cre(s) cre(a) des(b) des(c)
    if q == r:
        expectation += -2.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                        tpdm[s, :, :, :], optimize=True)
    #  (  -2.00000) k2(p,a,b,c) kdelta(q,r) cre(b) cre(c) des(s) des(a)
    if q == r:
        expectation += -2.0 * np.einsum('abc,bca', k2[p, :, :, :],
                                        tpdm[:, :, s, :], optimize=True)
    #  (   2.00000) k2(p,a,b,c) kdelta(q,s) cre(r) cre(a) des(b) des(c)
    if q == s:
        expectation += 2.0 * np.einsum('abc,abc', k2[p, :, :, :],
                                       tpdm[r, :, :, :], optimize=True)
    #  (   2.00000) k2(p,a,b,c) kdelta(q,s) cre(b) cre(c) des(r) des(a)
    if q == s:
        expectation += 2.0 * np.einsum('abc,bca', k2[p, :, :, :],
                                       tpdm[:, :, r, :], optimize=True)
    #  (   2.00000) k2(q,a,b,c) kdelta(p,r) cre(s) cre(a) des(b) des(c)
    if p == r:
        expectation += 2.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                       tpdm[s, :, :, :], optimize=True)
    #  (   2.00000) k2(q,a,b,c) kdelta(p,r) cre(b) cre(c) des(s) des(a)
    if p == r:
        expectation += 2.0 * np.einsum('abc,bca', k2[q, :, :, :],
                                       tpdm[:, :, s, :], optimize=True)
    #  (  -2.00000) k2(q,a,b,c) kdelta(p,s) cre(r) cre(a) des(b) des(c)
    if p == s:
        expectation += -2.0 * np.einsum('abc,abc', k2[q, :, :, :],
                                        tpdm[r, :, :, :], optimize=True)
    #  (  -2.00000) k2(q,a,b,c) kdelta(p,s) cre(b) cre(c) des(r) des(a)
    if p == s:
        expectation += -2.0 * np.einsum('abc,bca', k2[q, :, :, :],
                                        tpdm[:, :, r, :], optimize=True)
    return expectation


def k2_rotgen_hess_one_body(h1, p, q, r, s, opdm):
    """
    <| [[h1, p^ q - q^ p], r^s - s^r]|>

    This can be sped up with a dgemm call through numpy

    :param k2: reduced Hamiltonian in OpenFermion ordering. k2 shoudl be
               antisymmetric in pq and rs indices.
    :param p: spin-orbital index for p^q - q^p operator
    :param q: spin-orbital index for p^q - q^p operator
    :param p: spin-orbital index for r^s - s^r operator
    :param q: spin-orbital index for r^s - s^r operator
    :param tpdm: spin-orbital tpdm in OpenFermion ordering
    """
    expectation = 0.
    #  (   1.00000) h1(p,r) cre(q) des(s)
    expectation += 1.0 * opdm[q, s] * h1[p, r]
    #  (  -1.00000) h1(p,s) cre(q) des(r)
    expectation += -1.0 * opdm[q, r] * h1[p, s]
    #  (  -1.00000) h1(q,r) cre(p) des(s)
    expectation += -1.0 * opdm[p, s] * h1[q, r]
    #  (   1.00000) h1(q,s) cre(p) des(r)
    expectation += 1.0 * opdm[p, r] * h1[q, s]
    #  (   1.00000) h1(r,p) cre(s) des(q)
    expectation += 1.0 * opdm[s, q] * h1[r, p]
    #  (  -1.00000) h1(r,q) cre(s) des(p)
    expectation += -1.0 * opdm[s, p] * h1[r, q]
    #  (  -1.00000) h1(s,p) cre(r) des(q)
    expectation += -1.0 * opdm[r, q] * h1[s, p]
    #  (   1.00000) h1(s,q) cre(r) des(p)
    expectation += 1.0 * opdm[r, p] * h1[s, q]

    #  (   1.00000) h1(p,a) kdelta(q,r) cre(s) des(a)
    if q == r:
        expectation += 1.0 * np.einsum('a,a', h1[p, :], opdm[s, :])
    #  (  -1.00000) h1(p,a) kdelta(q,s) cre(r) des(a)
    if q == s:
        expectation += -1.0 * np.einsum('a,a', h1[p, :], opdm[r, :])
    #  (  -1.00000) h1(q,a) kdelta(p,r) cre(s) des(a)
    if p == r:
        expectation += -1.0 * np.einsum('a,a', h1[q, :], opdm[s, :])
    #  (   1.00000) h1(q,a) kdelta(p,s) cre(r) des(a)
    if p == s:
        expectation += 1.0 * np.einsum('a,a', h1[q, :], opdm[r, :])
    #  (   1.00000) h1(a,p) kdelta(q,r) cre(a) des(s)
    if q == r:
        expectation += 1.0 * np.einsum('a,a', h1[:, p], opdm[:, s])
    #  (  -1.00000) h1(a,p) kdelta(q,s) cre(a) des(r)
    if q == s:
        expectation += -1.0 * np.einsum('a,a', h1[:, p], opdm[:, r])
    #  (  -1.00000) h1(a,q) kdelta(p,r) cre(a) des(s)
    if p == r:
        expectation += -1.0 * np.einsum('a,a', h1[:, q], opdm[:, s])
    #  (   1.00000) h1(a,q) kdelta(p,s) cre(a) des(r)
    if p == s:
        expectation += 1.0 * np.einsum('a,a', h1[:, q], opdm[:, r])
    return expectation


def spinless_rotgrad_onebody(h1, p, q, sopdm):
    """
    Use spin-summed 1-RDM to get gradient with respect to the 1-body operartor

    h1 = \sum_{\sigma,mn}h_{mn}a_{m\sigma}^{\dagger}a_{n\sigma}

    <psi|[h1, E_{pq}]|psi>

    :param h1: one-electron integrals
    :param p: spatial index of E_{pq} - E_{qp}
    :param q: spatial index of E_{pq} - E_{qp}
    :param sopdm: spin-summed 1-RDM
    """
    expectation = 0.
    expectation += np.dot(h1[:, p], sopdm[:, q])
    expectation -= np.dot(h1[q, :], sopdm[p, :])
    expectation -= np.dot(h1[:, q], sopdm[:, p])
    expectation += np.dot(h1[p, :], sopdm[q, :])
    return expectation


def spinless_rothess_onebody(h1, p, q, r, s, sopdm):
    """
    Use spin-summed 1-RDM to get hessian with respect to the 1-body operartor

    h1 = \sum_{\sigma,mn}h_{mn}a_{m\sigma}^{\dagger}a_{n\sigma}

    <psi|[[h1, E_{pq}], E_{rs}]|psi>

    TODO:

    This can be simplified to save half the number of vec-vec products

    Even considering complex h1 and complex sopdm

    :param h1: one-electron integrals
    :param p: spatial index of E_{pq} - E_{qp}
    :param q: spatial index of E_{pq} - E_{qp}
    :param r: spatial index of E_{rs} - E_{rs}
    :param s: spatial index of E_{rs} - E_{rs}
    :param sopdm: spin-summed 1-RDM
    """
    expectation = 0.
    if r == q:
        expectation += np.dot(h1[:, p], sopdm[:, s])
    expectation -= h1[s, p] * sopdm[r, q]

    if s == q:
        expectation -= np.dot(h1[:, p], sopdm[:, r])
    expectation += h1[r, p] * sopdm[s, q]

    if p == s:
        expectation += np.dot(h1[q, :], sopdm[r, :])
    expectation -= h1[q, r] * sopdm[p, s]

    if p == r:
        expectation -= np.dot(h1[q, :], sopdm[s, :])
    expectation += h1[q, s] * sopdm[p, r]

    if p == r:
        expectation -= np.dot(h1[:, q], sopdm[:, s])
    expectation += h1[s, q] * sopdm[r, p]

    if p == s:
        expectation += np.dot(h1[:, q], sopdm[:, r])
    expectation -= h1[r, q] * sopdm[s, p]

    if q == s:
        expectation -= np.dot(h1[p, :], sopdm[r, :])
    expectation += h1[p, r] * sopdm[q, s]

    if q == r:
        expectation += np.dot(h1[p, :], sopdm[s, :])
    expectation -= h1[p, s] * sopdm[q, r]

    return expectation


def spinless_rotgrad_twobody(v2, m, n, stpdm):
    """
    Use spin-summed 2-RDM to get gradient with respect to the 2-body operator

    spin-summed 2-RDM

    e^{pq}_{rs} = \sum_{\sigma,\tau} <a_{p\sigma}^ a_{q\tau}^ a_{s\tau}a_{r\simga}>

    v2 = \sum_{\sigma,\tau,pqrs}V_{pqrs}a_{p\sigma}^ a_{q\tau}^ a_{s\tau}a_{r\simga}

    <psi|[v2, E_{mn}]|psi>

    :param v2: two-electron spatial integrals <1'2'|21>
    :param p: spatial index of E_{pq} - E_{qp}
    :param q: spatial index of E_{pq} - E_{qp}
    :param sopdm: spin-summed 1-RDM
    """
    expectation = 0.
    expectation -= np.einsum('qrs,qrs', v2[n, :, :, :], stpdm[m, :, :, :])
    expectation += np.einsum('qrs,qrs', v2[m, :, :, :], stpdm[n, :, :, :])

    expectation += np.einsum('prs,prs', v2[:, m, :, :], stpdm[:, n, :, :])
    expectation -= np.einsum('prs,prs', v2[:, n, :, :], stpdm[:, m, :, :])

    expectation -= np.einsum('pqs,pqs', v2[:, :, n, :], stpdm[:, :, m, :])
    expectation += np.einsum('pqs,pqs', v2[:, :, m, :], stpdm[:, :, n, :])

    expectation += np.einsum('pqr,pqr', v2[:, :, :, m], stpdm[:, :, :, n])
    expectation -= np.einsum('pqr,pqr', v2[:, :, :, n], stpdm[:, :, :, m])

    return expectation


def spinless_rothess_twobody(v2, m, n, t, u, stpdm):
    """
    spinless-hessian two-body terms

    Use spin-summed 2-RDM to get gradient with respect to the 2-body operator

    spin-summed 2-RDM

    e^{pq}_{rs} = \sum_{\sigma,\tau} <a_{p\sigma}^ a_{q\tau}^ a_{s\tau}a_{r\simga}>

    v2 = \sum_{\sigma,\tau,pqrs}V_{pqrs}a_{p\sigma}^ a_{q\tau}^ a_{s\tau}a_{r\simga}

    <psi|[[v2, E_{mn}], E_{tu}|psi>

    """
    expectation = 0.
    #   1.00000 v2(n, q, r, s) d(m,u) e(t,q,r,s)
    #   1.00000 v2(n, q, r, s) e(t,q,r,s)
    if u == m:
        expectation += np.einsum('qrs,qrs', v2[n, :, :, :], stpdm[t, :, :, :])

    #  -1.00000 v2(n, q, r, s) d(t,q) e(m,u,r,s)
    #  -1.00000 v2(n, t, r, s) e(m,u,r,s)
    expectation -= np.einsum('rs,rs', v2[n, t, :, :], stpdm[m, u, :, :])

    #   1.00000 v2(n, q, r, s) d(r,u) e(m,q,t,s)
    #   1.00000 v2(n, q, u, s) e(m,q,t,s)
    expectation += np.einsum('qs,qs', v2[n, :, u, :], stpdm[m, :, t, :])

    #  -1.00000 v2(n, q, r, s) d(t,s) e(m,q,r,u)
    #  -1.00000 v2(n, q, r, t) e(m,q,r,u)
    expectation -= np.einsum('qr,qr', v2[n, :, :, t], stpdm[m, :, :, u])

    #  -1.00000 v2(n, q, r, s) d(m,t) e(u,q,r,s)
    #  -1.00000 v2(n, q, r, s) e(u,q,r,s)
    if t == m:
        expectation -= np.einsum('qrs,qrs', v2[n, :, :, :], stpdm[u, :, :, :])

    #   1.00000 v2(n, q, r, s) d(u,q) e(m,t,r,s)
    #   1.00000 v2(n, u, r, s) e(m,t,r,s)
    expectation += np.einsum('rs,rs', v2[n, u, :, :], stpdm[m, t, :, :])

    #  -1.00000 v2(n, q, r, s) d(r,t) e(m,q,u,s)
    #  -1.00000 v2(n, q, t, s) e(m,q,u,s)
    expectation -= np.einsum('qs,qs', v2[n, :, t, :], stpdm[m, :, u, :])

    #   1.00000 v2(n, q, r, s) d(u,s) e(m,q,r,t)
    #   1.00000 v2(n, q, r, u) e(m,q,r,t)
    expectation += np.einsum('qr,qr', v2[n, :, :, u], stpdm[m, :, :, t])

    #  -1.00000 v2(p, m, r, s) d(p,u) e(t,n,r,s)
    #  -1.00000 v2(u, m, r, s) e(t,n,r,s)
    expectation -= np.einsum('rs,rs', v2[u, m, :, :], stpdm[t, n, :, :])

    #   1.00000 v2(p, m, r, s) d(t,n) e(p,u,r,s)
    #   1.00000 v2(p, m, r, s) e(p,u,r,s)
    if n == t:
        expectation += np.einsum('prs,prs', v2[:, m, :, :], stpdm[:, u, :, :])

    #  -1.00000 v2(p, m, r, s) d(r,u) e(p,n,t,s)
    #  -1.00000 v2(p, m, u, s) e(p,n,t,s)
    expectation -= np.einsum('ps,ps', v2[:, m, u, :], stpdm[:, n, t, :])

    #   1.00000 v2(p, m, r, s) d(t,s) e(p,n,r,u)
    #   1.00000 v2(p, m, r, t) e(p,n,r,u)
    expectation += np.einsum('pr,pr', v2[:, m, :, t], stpdm[:, n, :, u])

    #   1.00000 v2(p, m, r, s) d(p,t) e(u,n,r,s)
    #   1.00000 v2(t, m, r, s) e(u,n,r,s)
    expectation += np.einsum('rs,rs', v2[t, m, :, :], stpdm[u, n, :, :])

    #  -1.00000 v2(p, m, r, s) d(u,n) e(p,t,r,s)
    #  -1.00000 v2(p, m, r, s) e(p,t,r,s)
    if n == u:
        expectation -= np.einsum('prs,prs', v2[:, m, :, :], stpdm[:, t, :, :])

    #   1.00000 v2(p, m, r, s) d(r,t) e(p,n,u,s)
    #   1.00000 v2(p, m, t, s) e(p,n,u,s)
    expectation += np.einsum('ps,ps', v2[:, m, t, :], stpdm[:, n, u, :])

    #  -1.00000 v2(p, m, r, s) d(u,s) e(p,n,r,t)
    #  -1.00000 v2(p, m, r, u) e(p,n,r,t)
    expectation -= np.einsum('pr,pr', v2[:, m, :, u], stpdm[:, n, :, t])

    #   1.00000 v2(p, q, n, s) d(p,u) e(t,q,m,s)
    #   1.00000 v2(u, q, n, s) e(t,q,m,s)
    expectation += np.einsum('qs,qs', v2[u, :, n, :], stpdm[t, :, m, :])

    #  -1.00000 v2(p, q, n, s) d(t,q) e(p,u,m,s)
    #  -1.00000 v2(p, t, n, s) e(p,u,m,s)
    expectation -= np.einsum('ps,ps', v2[:, t, n, :], stpdm[:, u, m, :])

    #   1.00000 v2(p, q, n, s) d(m,u) e(p,q,t,s)
    #   1.00000 v2(p, q, n, s) e(p,q,t,s)
    if u == m:
        expectation += np.einsum('pqs,pqs', v2[:, :, n, :], stpdm[:, :, t, :])

    #  -1.00000 v2(p, q, n, s) d(t,s) e(p,q,m,u)
    #  -1.00000 v2(p, q, n, t) e(p,q,m,u)
    expectation -= np.einsum('pq,pq', v2[:, :, n, t], stpdm[:, :, m, u])

    #  -1.00000 v2(p, q, n, s) d(p,t) e(u,q,m,s)
    #  -1.00000 v2(t, q, n, s) e(u,q,m,s)
    expectation -= np.einsum('qs,qs', v2[t, :, n, :], stpdm[u, :, m, :])

    #   1.00000 v2(p, q, n, s) d(u,q) e(p,t,m,s)
    #   1.00000 v2(p, u, n, s) e(p,t,m,s)
    expectation += np.einsum('ps,ps', v2[:, u, n, :], stpdm[:, t, m, :])

    #  -1.00000 v2(p, q, n, s) d(m,t) e(p,q,u,s)
    #  -1.00000 v2(p, q, n, s) e(p,q,u,s)
    if t == m:
        expectation -= np.einsum('pqs,pqs', v2[:, :, n, :], stpdm[:, :, u, :])

    #   1.00000 v2(p, q, n, s) d(u,s) e(p,q,m,t)
    #   1.00000 v2(p, q, n, u) e(p,q,m,t)
    expectation += np.einsum('pq,pq', v2[:, :, n, u], stpdm[:, :, m, t])

    #  -1.00000 v2(p, q, r, m) d(p,u) e(t,q,r,n)
    #  -1.00000 v2(u, q, r, m) e(t,q,r,n)
    expectation -= np.einsum('qr,qr', v2[u, :, :, m], stpdm[t, :, :, n])

    #   1.00000 v2(p, q, r, m) d(t,q) e(p,u,r,n)
    #   1.00000 v2(p, t, r, m) e(p,u,r,n)
    expectation += np.einsum('pr,pr', v2[:, t, :, m], stpdm[:, u, :, n])

    #  -1.00000 v2(p, q, r, m) d(r,u) e(p,q,t,n)
    #  -1.00000 v2(p, q, u, m) e(p,q,t,n)
    expectation -= np.einsum('pq,pq', v2[:, :, u, m], stpdm[:, :, t, n])

    #   1.00000 v2(p, q, r, m) d(t,n) e(p,q,r,u)
    #   1.00000 v2(p, q, r, m) e(p,q,r,u)
    if n == t:
        expectation += np.einsum('pqr,pqr', v2[:, :, :, m], stpdm[:, :, :, u])

    #   1.00000 v2(p, q, r, m) d(p,t) e(u,q,r,n)
    #   1.00000 v2(t, q, r, m) e(u,q,r,n)
    expectation += np.einsum('qr,qr', v2[t, :, :, m], stpdm[u, :, :, n])

    #  -1.00000 v2(p, q, r, m) d(u,q) e(p,t,r,n)
    #  -1.00000 v2(p, u, r, m) e(p,t,r,n)
    expectation -= np.einsum('pr,pr', v2[:, u, :, m], stpdm[:, t, :, n])

    #   1.00000 v2(p, q, r, m) d(r,t) e(p,q,u,n)
    #   1.00000 v2(p, q, t, m) e(p,q,u,n)
    expectation += np.einsum('pq,pq', v2[:, :, t, m], stpdm[:, :, u, n])

    #  -1.00000 v2(p, q, r, m) d(u,n) e(p,q,r,t)
    #  -1.00000 v2(p, q, r, m) e(p,q,r,t)
    if n == u:
        expectation -= np.einsum('pqr,pqr', v2[:, :, :, m], stpdm[:, :, :, t])

    #  -1.00000 v2(m, q, r, s) d(n,u) e(t,q,r,s)
    #  -1.00000 v2(m, q, r, s) e(t,q,r,s)
    if u == n:
        expectation -= np.einsum('qrs,qrs', v2[m, :, :, :], stpdm[t, :, :, :])

    #   1.00000 v2(m, q, r, s) d(t,q) e(n,u,r,s)
    #   1.00000 v2(m, t, r, s) e(n,u,r,s)
    expectation += np.einsum('rs,rs', v2[m, t, :, :], stpdm[n, u, :, :])

    #  -1.00000 v2(m, q, r, s) d(r,u) e(n,q,t,s)
    #  -1.00000 v2(m, q, u, s) e(n,q,t,s)
    expectation -= np.einsum('qs,qs', v2[m, :, u, :], stpdm[n, :, t, :])

    #   1.00000 v2(m, q, r, s) d(t,s) e(n,q,r,u)
    #   1.00000 v2(m, q, r, t) e(n,q,r,u)
    expectation += np.einsum('qr,qr', v2[m, :, :, t], stpdm[n, :, :, u])

    #   1.00000 v2(m, q, r, s) d(n,t) e(u,q,r,s)
    #   1.00000 v2(m, q, r, s) e(u,q,r,s)
    if t == n:
        expectation += np.einsum('qrs,qrs', v2[m, :, :, :], stpdm[u, :, :, :])

    #  -1.00000 v2(m, q, r, s) d(u,q) e(n,t,r,s)
    #  -1.00000 v2(m, u, r, s) e(n,t,r,s)
    expectation -= np.einsum('rs,rs', v2[m, u, :, :], stpdm[n, t, :, :])

    #   1.00000 v2(m, q, r, s) d(r,t) e(n,q,u,s)
    #   1.00000 v2(m, q, t, s) e(n,q,u,s)
    expectation += np.einsum('qs,qs', v2[m, :, t, :], stpdm[n, :, u, :])

    #  -1.00000 v2(m, q, r, s) d(u,s) e(n,q,r,t)
    #  -1.00000 v2(m, q, r, u) e(n,q,r,t)
    expectation -= np.einsum('qr,qr', v2[m, :, :, u], stpdm[n, :, :, t])

    #   1.00000 v2(p, n, r, s) d(p,u) e(t,m,r,s)
    #   1.00000 v2(u, n, r, s) e(t,m,r,s)
    expectation += np.einsum('rs,rs', v2[u, n, :, :], stpdm[t, m, :, :])

    #  -1.00000 v2(p, n, r, s) d(t,m) e(p,u,r,s)
    #  -1.00000 v2(p, n, r, s) e(p,u,r,s)
    if m == t:
        expectation -= np.einsum('prs,prs', v2[:, n, :, :], stpdm[:, u, :, :])

    #   1.00000 v2(p, n, r, s) d(r,u) e(p,m,t,s)
    #   1.00000 v2(p, n, u, s) e(p,m,t,s)
    expectation += np.einsum('ps,ps', v2[:, n, u, :], stpdm[:, m, t, :])

    #  -1.00000 v2(p, n, r, s) d(t,s) e(p,m,r,u)
    #  -1.00000 v2(p, n, r, t) e(p,m,r,u)
    expectation -= np.einsum('pr,pr', v2[:, n, :, t], stpdm[:, m, :, u])

    #  -1.00000 v2(p, n, r, s) d(p,t) e(u,m,r,s)
    #  -1.00000 v2(t, n, r, s) e(u,m,r,s)
    expectation -= np.einsum('rs,rs', v2[t, n, :, :], stpdm[u, m, :, :])

    #   1.00000 v2(p, n, r, s) d(u,m) e(p,t,r,s)
    #   1.00000 v2(p, n, r, s) e(p,t,r,s)
    if m == u:
        expectation += np.einsum('prs,prs', v2[:, n, :, :], stpdm[:, t, :, :])

    #  -1.00000 v2(p, n, r, s) d(r,t) e(p,m,u,s)
    #  -1.00000 v2(p, n, t, s) e(p,m,u,s)
    expectation -= np.einsum('ps,ps', v2[:, n, t, :], stpdm[:, m, u, :])

    #   1.00000 v2(p, n, r, s) d(u,s) e(p,m,r,t)
    #   1.00000 v2(p, n, r, u) e(p,m,r,t)
    expectation += np.einsum('pr,pr', v2[:, n, :, u], stpdm[:, m, :, t])

    #  -1.00000 v2(p, q, m, s) d(p,u) e(t,q,n,s)
    #  -1.00000 v2(u, q, m, s) e(t,q,n,s)
    expectation -= np.einsum('qs,qs', v2[u, :, m, :], stpdm[t, :, n, :])

    #   1.00000 v2(p, q, m, s) d(t,q) e(p,u,n,s)
    #   1.00000 v2(p, t, m, s) e(p,u,n,s)
    expectation += np.einsum('ps,ps', v2[:, t, m, :], stpdm[:, u, n, :])

    #  -1.00000 v2(p, q, m, s) d(n,u) e(p,q,t,s)
    #  -1.00000 v2(p, q, m, s) e(p,q,t,s)
    if u == n:
        expectation -= np.einsum('pqs,pqs', v2[:, :, m, :], stpdm[:, :, t, :])

    #   1.00000 v2(p, q, m, s) d(t,s) e(p,q,n,u)
    #   1.00000 v2(p, q, m, t) e(p,q,n,u)
    expectation += np.einsum('pq,pq', v2[:, :, m, t], stpdm[:, :, n, u])

    #   1.00000 v2(p, q, m, s) d(p,t) e(u,q,n,s)
    #   1.00000 v2(t, q, m, s) e(u,q,n,s)
    expectation += np.einsum('qs,qs', v2[t, :, m, :], stpdm[u, :, n, :])

    #  -1.00000 v2(p, q, m, s) d(u,q) e(p,t,n,s)
    #  -1.00000 v2(p, u, m, s) e(p,t,n,s)
    expectation -= np.einsum('ps,ps', v2[:, u, m, :], stpdm[:, t, n, :])

    #   1.00000 v2(p, q, m, s) d(n,t) e(p,q,u,s)
    #   1.00000 v2(p, q, m, s) e(p,q,u,s)
    if t == n:
        expectation += np.einsum('pqs,pqs', v2[:, :, m, :], stpdm[:, :, u, :])

    #  -1.00000 v2(p, q, m, s) d(u,s) e(p,q,n,t)
    #  -1.00000 v2(p, q, m, u) e(p,q,n,t)
    expectation -= np.einsum('pq,pq', v2[:, :, m, u], stpdm[:, :, n, t])

    #   1.00000 v2(p, q, r, n) d(p,u) e(t,q,r,m)
    #   1.00000 v2(u, q, r, n) e(t,q,r,m)
    expectation += np.einsum('qr,qr', v2[u, :, :, n], stpdm[t, :, :, m])

    #  -1.00000 v2(p, q, r, n) d(t,q) e(p,u,r,m)
    #  -1.00000 v2(p, t, r, n) e(p,u,r,m)
    expectation -= np.einsum('pr,pr', v2[:, t, :, n], stpdm[:, u, :, m])

    #   1.00000 v2(p, q, r, n) d(r,u) e(p,q,t,m)
    #   1.00000 v2(p, q, u, n) e(p,q,t,m)
    expectation += np.einsum('pq,pq', v2[:, :, u, n], stpdm[:, :, t, m])

    #  -1.00000 v2(p, q, r, n) d(t,m) e(p,q,r,u)
    #  -1.00000 v2(p, q, r, n) e(p,q,r,u)
    if m == t:
        expectation -= np.einsum('pqr,pqr', v2[:, :, :, n], stpdm[:, :, :, u])

    #  -1.00000 v2(p, q, r, n) d(p,t) e(u,q,r,m)
    #  -1.00000 v2(t, q, r, n) e(u,q,r,m)
    expectation -= np.einsum('qr,qr', v2[t, :, :, n], stpdm[u, :, :, m])

    #   1.00000 v2(p, q, r, n) d(u,q) e(p,t,r,m)
    #   1.00000 v2(p, u, r, n) e(p,t,r,m)
    expectation += np.einsum('pr,pr', v2[:, u, :, n], stpdm[:, t, :, m])

    #  -1.00000 v2(p, q, r, n) d(r,t) e(p,q,u,m)
    #  -1.00000 v2(p, q, t, n) e(p,q,u,m)
    expectation -= np.einsum('pq,pq', v2[:, :, t, n], stpdm[:, :, u, m])

    #   1.00000 v2(p, q, r, n) d(u,m) e(p,q,r,t)
    #   1.00000 v2(p, q, r, n) e(p,q,r,t)
    if m == u:
        expectation += np.einsum('pqr,pqr', v2[:, :, :, n], stpdm[:, :, :, t])
    return expectation
