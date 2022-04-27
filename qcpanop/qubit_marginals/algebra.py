# Much of this code is borrowed from Nik Tezak's implementation in Grove.
from collections import OrderedDict
import itertools

import numpy as np


class OperatorBasis(object):
    """
    Encapsulate a complete set of basis operators.
    """

    def __init__(self, labels_ops):
        """
        Encapsulates a set of linearly independent operators.

        :param (list|tuple) labels_ops: Sequence of tuples (label, operator) where label is a string
            and operator is a numpy.ndarray/
        """
        self.ops_by_label = OrderedDict(labels_ops)
        self.labels = list(self.ops_by_label.keys())
        self.ops = list(self.ops_by_label.values())
        self.dim = len(self.ops)

    def product(self, *bases):
        """
        Compute the tensor product with another basis.

        :param bases: One or more additional bases to form the product with.
        :return (OperatorBasis): The tensor product basis as an OperatorBasis object.
        """
        if len(bases) > 1:
            basis_rest = bases[0].product(*bases[1:])
        else:
            assert len(bases) == 1
            basis_rest = bases[0]

        labels_ops = [(b1l + b2l, np.kron(b1, b2)) for (b1l, b1), (b2l, b2) in
                      itertools.product(self, basis_rest)]

        return OperatorBasis(labels_ops)

    def __iter__(self):
        """
        Iterate over tuples of (label, basis_op)

        :return: Yields the labels and qutip operators corresponding to the vectors in this basis.
        :rtype: tuple (str, qutip.qobj.Qobj)
        """
        for l, op in zip(self.labels, self.ops):
            yield l, op

    def __pow__(self, n):
        """
        Create the n-fold tensor product basis.

        :param int n: The number of identical tensor factors.
        :return: The product basis.
        :rtype: OperatorBasis
        """
        if not isinstance(n, int):
            raise TypeError("Can only accept an integer number of factors")
        if n < 1:
            raise ValueError("Need positive number of factors")
        if n == 1:
            return self
        return self.product(*([self] * (n - 1)))

    def __repr__(self):
        return "<span[{}]>".format(",".join(self.labels))