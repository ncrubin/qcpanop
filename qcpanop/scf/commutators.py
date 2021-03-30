import numpy as np


def k2_rotgen_grad_one_body(h1, p, q, opdm):
    """
    [h_{pq}, pq]
    """
    expectation = 0.
    # #  (   1.00000) h1(p,a) cre(q) des(a)
    # #  (   1.00000) h1(a,p) cre(a) des(q)
    # expectation += np.dot(h1[p, :], opdm[q, :]) + np.dot(h1[:, p], opdm[:, q])
    # #  (  -1.00000) h1(q,a) cre(p) des(a)
    # #  (  -1.00000) h1(a,q) cre(a) des(p)
    # expectation -= np.dot(h1[q, :], opdm[p, :]) + np.dot(h1[:, q], opdm[:, p])
    # return expectation

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

    This can probably sped up with blas call to vector dot on reshaped and
    flattened k2 and tpdm
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

    # Contraction from NCR's code
    expectation2 = 0.
    #   1.00000 h1(m, n) E(m,q) d(p,n)
    #   1.00000 h1(m, p) E(m,q)
    expectation2 += np.einsum('m,m', h1[:, p], sopdm[:, q])

    #  -1.00000 h1(m, n) E(p,n) d(m,q)
    #  -1.00000 h1(q, n) E(p,n)
    expectation2 -= np.einsum('n,n', h1[q, :], sopdm[p, :])

    #  -1.00000 h1(m, n) E(m,p) d(q,n)
    #  -1.00000 h1(m, q) E(m,p)
    expectation2 -= np.einsum('m,m', h1[:, q], sopdm[:, p])

    #   1.00000 h1(m, n) E(q,n) d(m,p)
    #   1.00000 h1(p, n) E(q,n)
    expectation2 += np.einsum('n,n', h1[p, :], sopdm[q, :])
    assert np.isclose(expectation2, expectation)
    return expectation


def spinless_rothess_onebody(h1, p, q, r, s, sopdm):
    """
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


    # Auto Generated from NCR's tools.
    expectation2 = 0.
    #   1.00000 h1(m, p) E(m,s) d(r,q)
    #   1.00000 h1(m, p) E(m,s)
    if q == r:
        expectation2 += np.einsum('m,m', h1[:, p], sopdm[:, s])

    #  -1.00000 h1(m, p) E(r,q) d(m,s)
    #  -1.00000 h1(s, p) E(r,q)
    expectation2 += -1.00000 * h1[s, p] * sopdm[r, q]

    #  -1.00000 h1(m, p) E(m,r) d(s,q)
    #  -1.00000 h1(m, p) E(m,r)
    if q == s:
        expectation2 -= np.einsum('m,m', h1[:, p], sopdm[:, r])

    #   1.00000 h1(m, p) E(s,q) d(m,r)
    #   1.00000 h1(r, p) E(s,q)
    expectation2 += 1.00000 * h1[r, p] * sopdm[s, q]

    #  -1.00000 h1(q, n) E(p,s) d(r,n)
    #  -1.00000 h1(q, r) E(p,s)
    expectation2 += -1.00000 * h1[q, r] * sopdm[p, s]

    #   1.00000 h1(q, n) E(r,n) d(p,s)
    #   1.00000 h1(q, n) E(r,n)
    if s == p:
        expectation2 += np.einsum('n,n', h1[q, :], sopdm[r, :])

    #   1.00000 h1(q, n) E(p,r) d(s,n)
    #   1.00000 h1(q, s) E(p,r)
    expectation2 += 1.00000 * h1[q, s] * sopdm[p, r]

    #  -1.00000 h1(q, n) E(s,n) d(p,r)
    #  -1.00000 h1(q, n) E(s,n)
    if r == p:
        expectation2 -= np.einsum('n,n', h1[q, :], sopdm[s, :])

    #  -1.00000 h1(m, q) E(m,s) d(r,p)
    #  -1.00000 h1(m, q) E(m,s)
    if p == r:
        expectation2 -= np.einsum('m,m', h1[:, q], sopdm[:, s])

    #   1.00000 h1(m, q) E(r,p) d(m,s)
    #   1.00000 h1(s, q) E(r,p)
    expectation2 += 1.00000 * h1[s, q] * sopdm[r, p]

    #   1.00000 h1(m, q) E(m,r) d(s,p)
    #   1.00000 h1(m, q) E(m,r)
    if p == s:
        expectation2 += np.einsum('m,m', h1[:, q], sopdm[:, r])

    #  -1.00000 h1(m, q) E(s,p) d(m,r)
    #  -1.00000 h1(r, q) E(s,p)
    expectation2 += -1.00000 * h1[r, q] * sopdm[s, p]

    #   1.00000 h1(p, n) E(q,s) d(r,n)
    #   1.00000 h1(p, r) E(q,s)
    expectation2 += 1.00000 * h1[p, r] * sopdm[q, s]

    #  -1.00000 h1(p, n) E(r,n) d(q,s)
    #  -1.00000 h1(p, n) E(r,n)
    if s == q:
        expectation2 -= np.einsum('n,n', h1[p, :], sopdm[r, :])

    #  -1.00000 h1(p, n) E(q,r) d(s,n)
    #  -1.00000 h1(p, s) E(q,r)
    expectation2 += -1.00000 * h1[p, s] * sopdm[q, r]

    #   1.00000 h1(p, n) E(s,n) d(q,r)
    #   1.00000 h1(p, n) E(s,n)
    if r == q:
        expectation2 += np.einsum('n,n', h1[p, :], sopdm[s, :])

    assert np.isclose(expectation2, expectation)
    return expectation


def spinless_rotgrad_twobody(v2, m, n, stpdm):
    """
    [v2e_{pqrs}, E_{mn} -E_{nm}]
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

    # Contract from NCR's code
    expectation2 = 0.
    #  -1.00000 v2(p, q, r, s) d(p,n) e(m,q,r,s)
    #  -1.00000 v2(n, q, r, s) e(m,q,r,s)
    expectation2 -= np.einsum('qrs,qrs', v2[n, :, :, :], stpdm[m, :, :, :])

    #   1.00000 v2(p, q, r, s) d(m,q) e(p,n,r,s)
    #   1.00000 v2(p, m, r, s) e(p,n,r,s)
    expectation2 += np.einsum('prs,prs', v2[:, m, :, :], stpdm[:, n, :, :])

    #  -1.00000 v2(p, q, r, s) d(r,n) e(p,q,m,s)
    #  -1.00000 v2(p, q, n, s) e(p,q,m,s)
    expectation2 -= np.einsum('pqs,pqs', v2[:, :, n, :], stpdm[:, :, m, :])

    #   1.00000 v2(p, q, r, s) d(m,s) e(p,q,r,n)
    #   1.00000 v2(p, q, r, m) e(p,q,r,n)
    expectation2 += np.einsum('pqr,pqr', v2[:, :, :, m], stpdm[:, :, :, n])

    #   1.00000 v2(p, q, r, s) d(p,m) e(n,q,r,s)
    #   1.00000 v2(m, q, r, s) e(n,q,r,s)
    expectation2 += np.einsum('qrs,qrs', v2[m, :, :, :], stpdm[n, :, :, :])

    #  -1.00000 v2(p, q, r, s) d(n,q) e(p,m,r,s)
    #  -1.00000 v2(p, n, r, s) e(p,m,r,s)
    expectation2 -= np.einsum('prs,prs', v2[:, n, :, :], stpdm[:, m, :, :])

    #   1.00000 v2(p, q, r, s) d(r,m) e(p,q,n,s)
    #   1.00000 v2(p, q, m, s) e(p,q,n,s)
    expectation2 += np.einsum('pqs,pqs', v2[:, :, m, :], stpdm[:, :, n, :])

    #  -1.00000 v2(p, q, r, s) d(n,s) e(p,q,r,m)
    #  -1.00000 v2(p, q, r, n) e(p,q,r,m)
    expectation2 -= np.einsum('pqr,pqr', v2[:, :, :, n], stpdm[:, :, :, m])

    assert np.isclose(expectation2, expectation)

    return expectation


def spinless_rothess_twobody(v2, m, n, t, u, stpdm):
    """
    spinless-hessian two-body terms
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
