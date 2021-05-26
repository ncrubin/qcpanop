"""
DIIS object
"""
import numpy as np
from itertools import product


class DIIS:

    def __init__(self, num_diis_vecs, start_vecs=2):
        self.nvecs = num_diis_vecs
        self.error_vecs = []
        self.p_vecs = []
        self.start_vecs = start_vecs

    def compute_new_vec(self, p, error):
        if len(self.p_vecs) < self.start_vecs:
            self.p_vecs.append(p)
            self.error_vecs.append(error)
            return p

        self.p_vecs.append(p)
        self.error_vecs.append(error)
        if len(self.p_vecs) > self.nvecs:
            self.p_vecs.pop(0)
            self.error_vecs.pop(0)

        b_mat, rhs = self.get_bmat()
        c = np.linalg.solve(b_mat, rhs)
        new_p = np.zeros_like(self.p_vecs[0])
        for ii in range(len(self.p_vecs)):
            new_p += c[ii] * self.p_vecs[ii]
        return new_p

    def get_bmat(self):
        """
        Compute b-mat
        """
        dim = len(self.p_vecs)
        b = np.zeros((dim, dim))
        for i, j in product(range(dim), repeat=2):
            if i <= j:
                b[i, j] = self.edot(self.error_vecs[i], self.error_vecs[j])
                b[j, i] = b[i, j]
        b = np.hstack((b, -1 * np.ones((dim, 1))))
        b = np.vstack((b, -1 * np.ones((1, dim + 1))))
        b[-1, -1] = 0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1, 0] = -1
        return b, rhs

    def edot(self, e1, e2):
        """
        e1 and e2 aren't necessarily vectors. If matrices do a matrix dot
        :param e1: error vec1
        :param e2: erorr vec2
        """
        if len(e1.shape) == 1 and len(e2.shape) == 1:
            return e1.dot(e2)
        elif e1.shape[1] == 1 and e2.shape[1] == 1:
            return e1.T.dot(e2)
        elif len(e1.shape) == 2 and len(e2.shape) == 2 and e1.shape == e2.shape:
            return np.einsum('ij,ij', e1, e2)  # Tr[e1.T @ e2]
        else:
            raise TypeError("Can't take dot of this type of error vec")
