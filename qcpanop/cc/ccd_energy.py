
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
# h
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
