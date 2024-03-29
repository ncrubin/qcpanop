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
        return "G({},{},{},{})".format(*self.indices)


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


def commutator_obop(op1, op2):

    term1 = Term(1 * op1.coeff * op2.coeff, [OneBodyOp(op1.up, op2.dwn), KroneckerDelta(op2.up, op1.dwn)])
    term2 = Term(-1 * op1.coeff * op2.coeff, [OneBodyOp(op2.up, op1.dwn), KroneckerDelta(op1.up, op2.dwn)])
    return [term1, term2]


def commutator_tbob(tb1, ob2):
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


def get_constant_terms(term, constants):
    constant_terms = []
    contractable = []
    for tt in term:
        if all([xx in constants for xx in tt.indices]):
            constant_terms.append(tt)
        else:
            contractable.append(tt)
    return constant_terms, contractable


def get_deltas(terms):
    non_deltas = []
    deltas = []
    for tt in terms:
        if isinstance(tt, KroneckerDelta):
            deltas.append(tt)
        else:
            non_deltas.append(tt)
    return deltas, non_deltas


def get_delta_map(term, summed_indices, constant_indices):
    deltas, _ = get_deltas(term)
    delta_map = {}
    for dd in deltas:
        summed_idx = dd.indices[0] if dd.indices[0] in summed_indices else dd.indices[1]
        mapped_idx = dd.indices[0] if dd.indices[0] in constant_indices else dd.indices[1]
        delta_map[summed_idx] = mapped_idx
    return delta_map


def get_new_term_with_subbed_indices(old_term, index_map):
    new_term_list = []
    for tt in old_term:
        new_indices = [index_map[xx] if xx in index_map.keys() else xx for xx in tt.indices]

        if isinstance(tt, Tensor):
            new_term_list.append(Tensor(tt.name, new_indices))
        elif isinstance(tt, OneBodyOp):
            new_term_list.append(OneBodyOp(new_indices[0], new_indices[1],
                                           coeff=tt.coeff))
        elif isinstance(tt, TwoBodyOp):
            new_term_list.append(TwoBodyOp(new_indices[0], new_indices[1],
                                           new_indices[2], new_indices[3],
                                           coeff=tt.coeff))
    new_term = Term(old_term.cnst, new_term_list)
    return new_term


def contract_one_body(term_list, summed_indices=None, constant_indices=None):
    contracted_terms = []
    for term in term_list:
        print("# ", term)
        delta_map = get_delta_map(term, summed_indices, constant_indices)
        new_term = get_new_term_with_subbed_indices(term, delta_map)
        print("# ", new_term)
        contracted_terms.append(new_term)


        einsum_string = ""
        if all([xx in constant_indices for xx in delta_map.keys()]) and all([xx in constant_indices for xx in delta_map.values()]):
            einsum_string = "if {} == {}:\n\t".format(list(delta_map.keys())[0], list(delta_map.values())[0])

        if all([xx in constant_indices for xx in new_term.get_indices()]):
            einsum_string += "expectation2 += {: 5.5f} ".format(new_term.cnst)
            op_names = []
            op_strings = []
            contracted_indices = []
            for ops in new_term.terms:
                op_names.append(ops.name)
                current_op_string = []
                contracted_index = []
                for oidx in ops.indices:
                    if oidx in constant_indices:
                        current_op_string.append(oidx)
                    else:
                        current_op_string.append(':')
                        contracted_index.append(oidx)
                op_strings.append(current_op_string)
                contracted_indices.append(contracted_index)

            for oname, oindexer in zip(op_names, op_strings):
                einsum_string += "* {}[".format(oname) + ", ".join(
                    oindexer) + "] "
            print(einsum_string)
            print()
            continue

        einsum_string += "expectation2 "
        einsum_string += "-=" if np.sign(new_term.cnst) < 0 else "+="
        einsum_string += " np.einsum(" if np.isclose(np.abs(new_term.cnst), 1) else " {} * np.einsum(".format(new_term.cnst)
        op_names = []
        op_strings = []
        contracted_indices = []
        for ops in new_term.terms:
            op_names.append(ops.name)
            current_op_string = []
            contracted_index = []
            for oidx in ops.indices:
                if oidx in constant_indices:
                    current_op_string.append(oidx)
                else:
                    current_op_string.append(':')
                    contracted_index.append(oidx)
            op_strings.append(current_op_string)
            contracted_indices.append(contracted_index)

        einsum_string += "\'" + ",".join("".join(xx) for xx in contracted_indices) + "\'"
        for oname, oindexer in zip(op_names, op_strings):
            einsum_string += ", {}[".format(oname) + ", ".join(oindexer) + "]"
        einsum_string += ")"
        print(einsum_string)
        print()
    return contracted_terms


def contract_two_body(term_list, summed_indices=None, constant_indices=None):
    contracted_terms = []
    for term in term_list:
        print("# ", term)
        delta_map = get_delta_map(term, summed_indices, constant_indices)
        new_term = get_new_term_with_subbed_indices(term, delta_map)
        print("# ", new_term)
        contracted_terms.append(new_term)


def get_excitation_term(term: Term):
    ex_term = None
    new_term = Term(term.cnst, [])
    for tt in term.terms:
        if isinstance(tt, (OneBodyOp, TwoBodyOp)):
            ex_term = copy.deepcopy(tt)
        else:
            new_term.terms.append(tt)
    return ex_term, new_term


def get_tensor_term(term: Term):
    ex_term = None
    new_term = Term(term.cnst, [])
    for tt in term.terms:
        if isinstance(tt, (Tensor)):
            ex_term = copy.deepcopy(tt)
        else:
            new_term.terms.append(tt)
    return ex_term, new_term


if __name__ == "__main__":

    kpq = Tensor('k', ['p', 'q'])
    krs = Tensor('k', ['r', 's'])
    t2 = Tensor('tbt', ['i', 'j', 'k', 'l'])
    e_ijkl = TwoBodyOp('i', 'j', 'k', 'l')
    E_pq = OneBodyOp('p', 'q')
    E_rs = OneBodyOp('r', 's')
    print("sum_pq [{},{}]".format(e_ijkl, E_pq))
    terms = commutator_tbob(e_ijkl, E_pq)
    terms_with_contraction = []
    contracted_pq = []
    for tt in terms:
        tt.terms.insert(0, kpq)
        terms_with_contraction.append(tt)
        print(terms_with_contraction[-1])
        delta_map = get_delta_map(terms_with_contraction[-1],
                                  ['p', 'q'],
                                  ['i', 'j', 'k', 'l'])
        new_term = get_new_term_with_subbed_indices(
            terms_with_contraction[-1], delta_map)
        contracted_pq.append(new_term)
        print(new_term)
        print()

    double_comm = []
    for tt in contracted_pq:
        excitation, non_excitation = get_excitation_term(tt)
        print(non_excitation, "\t", excitation)

        print("Calculating [{},{}]".format(excitation, E_rs))
        new_double_comm = commutator_tbob(excitation, E_rs)
        for ss in new_double_comm:
            ss.terms.insert(0, krs)
            print(ss)
            delta_map = get_delta_map(ss,
                                      ['r', 's'],
                                      ['i', 'j', 'k', 'l', 'p', 'q'])
            new_term = get_new_term_with_subbed_indices(
                ss, delta_map)
            print(new_term)
            double_comm.append(Term(new_term.cnst * non_excitation.cnst,
                                    non_excitation.terms + new_term.terms))
            print(double_comm[-1])
            print('-------------------')
        print()

    print("Full double comm")
    for dd in double_comm:
        print(dd)

    print("\n\nd T/(d Kab)")
    temp_grad_vals = []
    for dd in double_comm:
        print(dd)
        excitation, non_excitation = get_excitation_term(dd)
        k1, k2 = non_excitation.terms
        temp_grad_vals.append(Term(dd.cnst, [KroneckerDelta(k1.indices[0], 'a'), KroneckerDelta(k1.indices[1], 'b'), k2, excitation]))
        print(temp_grad_vals[-1])
        temp_grad_vals.append(Term(dd.cnst, [KroneckerDelta(k2.indices[0], 'a'), KroneckerDelta(k2.indices[1], 'b'), k1, excitation]))
        print(temp_grad_vals[-1])
        print()

    print("\n\nd T/ (d Kcd d Kab)")
    hess_vals = []
    for dt in temp_grad_vals:
        print(dt)
        tensor_term, non_tensor_term = get_tensor_term(dt)

        non_tensor_term.terms.insert(0, KroneckerDelta(tensor_term.indices[1], 'd'))
        non_tensor_term.terms.insert(0, KroneckerDelta(tensor_term.indices[0], 'c'))
        non_tensor_term.terms.insert(0, t2)
        print(non_tensor_term)
        delta_map = get_delta_map(non_tensor_term,
                                  ['i', 'j', 'k', 'l', 'p', 'q', 'r', 's'],
                                  ['a', 'b', 'c', 'd'])
        new_term = get_new_term_with_subbed_indices(
            non_tensor_term, delta_map)
        print(new_term)
        print('------------------')











