
import sys
sys.path.insert(0, './..')

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("")
ahat.set_print_level(0)

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

# one-electron part:
ahat.add_operator_product(1.0,['h(p,q)'])

# [h, T2]
ahat.add_commutator(1.0,['h(p,q)','t2(a,b,i,j)'])


# two-electron part:

# g
ahat.add_operator_product(1.0,['g(p,r,q,s)'])

# [g, T2]
ahat.add_commutator(1.0,['g(p,r,q,s)','t2(a,b,i,j)'])

ahat.simplify()

ahat.print_fully_contracted()

ahat.clear()

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("doubles")
ahat.set_print_level(0)

print('')
print('    < 0 | m* n* f e e(-T) H e(T) | 0> :')
print('')

# needs single commutator
ahat.add_operator_product(1.0, ['h(p,q)'])

# [h, T2]
ahat.add_commutator(1.0,['h(p,q)','t2(a,b,i,j)'])

# # two-electron part: need up to double comm.
#
# g
ahat.add_operator_product(1.0,['g(p,r,q,s)'])
#
# [g, T2]
ahat.add_commutator(1.0,['g(p,r,q,s)','t2(a,b,i,j)'])

# [[g, T2, T2]]
ahat.add_double_commutator( 0.5, ['g(p,r,q,s)','t2(a,b,i,j)','t2(c,d,k,l)'])


ahat.simplify()

ahat.print_fully_contracted()

ahat.clear()

