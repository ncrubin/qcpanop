"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory

"""
import pdaggerq

import numpy as np

import openfermion as of

def singles():
    print("\n\n\n")
    print("CCS Commutator")
    pq = pdaggerq.pq_helper('true')
    pq.add_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                           ['a*(m)', 'a(n)'],
                                  )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        print(tt)
    pq.clear()
    print()
    print()
    print()
    print("double commutator")

    pq = pdaggerq.pq_helper('true')
    pq.add_double_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                           ['a*(m)', 'a(n)'],
                           ['a*(m)', 'a(n)'],
                                  )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        unique_terms = np.unique(tt)
        if len(unique_terms) == len(tt):
            print(tt)
        else:
            continue
    pq.clear()


    print("\n\n\n\nTriple commutator")
    pq = pdaggerq.pq_helper('true')
    pq.add_triple_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                                  ['a*(m)', 'a(n)'],
                                  ['a*(m)', 'a(n)'],
                                  ['a*(m)', 'a(n)'],
                                   )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        unique_terms = np.unique(tt)
        if len(unique_terms) == len(tt):
            print(tt)
        else:
            continue
    pq.clear()


def doubles():
    print("\n\n\n")
    print("CCD Commutator")
    pq = pdaggerq.pq_helper('true')
    pq.add_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                           ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                                  )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        print(tt)
    pq.clear()
    print()
    print()
    print()
    print("double commutator")

    pq = pdaggerq.pq_helper('true')
    pq.add_double_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                           ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                           ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                                  )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        unique_terms = np.unique(tt)
        if len(unique_terms) == len(tt):
            print(tt)
        else:
            continue
    pq.clear()


    print("\n\n\n\nTriple commutator")
    pq = pdaggerq.pq_helper('true')
    pq.add_triple_commutator(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'],
                                  ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                                  ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                                  ['a*(m)', 'a(n)', 'a*(o)', 'a(p)'],
                            )
    pq.simplify()
    # pq.print(string_type = 'all')
    vals = pq.strings()
    for tt in vals:
        unique_terms = np.unique(tt)
        if len(unique_terms) == len(tt):
            print(tt)
        else:
            continue
    pq.clear()



if __name__ == "__main__":
    singles()
    doubles()