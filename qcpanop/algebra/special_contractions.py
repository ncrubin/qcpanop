import numpy as np
import copy

class KroneckerDelta:
    def __init__(self, i, j):
        self.indices = [i, j]
        self.name = 'kd'

    def __repr__(self):
        return "d({},{})".format(self.indices[0], self.indices[1])


class OneBodyOp:
    def __init__(self, m, n, coeff=1.):
        self.coeff = coeff
        self.indices = [m, n]
        self.up = self.indices[0]
        self.dwn = self.indices[1]
        self.m = self.indices[0]
        self.n = self.indices[1]
        self.name = 'sopdm'

    def __repr__(self):
        return "E({},{})".format(self.up, self.dwn)


class TwoBodyOp:
    def __init__(self, p, q, r, s, coeff=1.):
        self.coeff = coeff
        self.indices = [p, q, r, s]
        self.p = self.indices[0]
        self.q = self.indices[1]
        self.r = self.indices[2]
        self.s = self.indices[3]
        self.name = 'stpdm'

    def __repr__(self):
        return "e({},{},{},{})".format(*self.indices)


class Term:

    def __init__(self, cnst, operators):
        """a collection of terms"""
        self.cnst = cnst
        self.terms = operators

    def __repr__(self):
        output = "{: 5.5f} ".format(self.cnst)
        for tt in self.terms:
            output += tt.__repr__() + " "
        return output

    def __iter__(self):
        for tt in self.terms:
            yield tt

    def get_indices(self):
        return list(np.unique([xx.indices for xx in self.terms]))


class Tensor:

    def __init__(self, name, indices):
        self.indices = indices
        self.name = name

    def __repr__(self):
        output = "{}(".format(self.name) + ", ".join(self.indices) + ")"
        return output



def commutator_two_body_one_body(tb1, ob2):
    m, n = ob2.m, ob2.n
    p, q, r, s = tb1.indices
    sign1 = -1 * tb1.coeff * ob2.coeff
    term1 = [KroneckerDelta(p, n), TwoBodyOp(m, q, r, s)]
    t1 = Term(sign1, term1)
    sign2 = 1 * tb1.coeff * ob2.coeff
    term2 = [KroneckerDelta(m, q), TwoBodyOp(p, n, r, s)]
    t2 = Term(sign2, term2)
    sign3 = -1 * tb1.coeff * ob2.coeff
    term3 = [KroneckerDelta(r, n), TwoBodyOp(p, q, m, s)]
    t3 = Term(sign3, term3)
    sign4 = 1 * tb1.coeff * ob2.coeff
    term4 = [KroneckerDelta(m, s), TwoBodyOp(p, q, r, n)]
    t4 = Term(sign4, term4)
    return [t1, t2, t3, t4]


def main():
    e_ijkl = TwoBodyOp('i', 'j', 'k', 'l')
    e_lkji = TwoBodyOp('l', 'k', 'j', 'i')
    E_pq = OneBodyOp('p', 'q')
    E_qp = OneBodyOp('p', 'q', coeff=-1.)
    terms = commutator_two_body_one_body(e_ijkl, E_pq)
    for tt in terms:
        print(tt)


if __name__ == "__main__":
    main()