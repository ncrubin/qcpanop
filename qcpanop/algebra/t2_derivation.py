"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

import openfermion as of

def main():

    print("\n\n\n")
    print("T2 mappings - anticommutator {b_{ijk}, b_{lmn}*}")
    pq = pdaggerq.pq_helper('true')
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a(k)', 'a*(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a*(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a(k)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()

    print("\n\n\n")
    print("T2 David-mappings - anticommutator {b_{ijk}, b_{lmn}*}")
    pq = pdaggerq.pq_helper('true')
    pq.add_operator_product(1.0, ['a*(j)', 'a*(k)', 'a(l)', 'a*(q)', 'a(p)', 'a(n)'])
    pq.add_operator_product(1.0, ['a(j)', 'a(k)', 'a*(l)', 'a(q)', 'a*(p)', 'a*(n)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()


    print("T1 mappings")
    # T1 = {a*(i)a*(j)a*(k), a(n)a(m)a(l)}
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a*(k)', 'a(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a*(k)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()


    print("Generalized T2")
    # {a*(i)a*(j)a(k) + a*(l), (a*(m)a*(n)a(p) + a*(q))*}
    # {a*(i)a*(j)a(k) + a*(l), a*(p)a(n)a(m) + a(q)}

    # (a*(i)a*(j)a(k) + a*(l))(a*(p)a(n)a(m) + a(q)) + (a*(p)a(n)a(m) + a(q))(a*(i)a*(j)a(k) + a*(l))

    # a*(i)a*(j)a(k)a*(p)a(n)a(m) + a*(i)a*(j)a(k)a(q) + a*(l)a*(p)a(n)a(m) + a*(l)a(q)
    # a*(p)a(n)a(m)a*(i)a*(j)a(k) + a*(p)a(n)a(m)a*(l) + a(q)a*(i)a*(j)a(k) + a(q)a*(l)

 
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a(k)', 'a*(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a*(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a(k)'])
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a(k)', 'a(q)'])
    pq.add_operator_product(1.0, ['a*(p)', 'a(n)', 'a(m)', 'a*(l)'])
    pq.add_operator_product(1.0, ['a*(l)', 'a*(p)', 'a(n)', 'a(m)'])
    pq.add_operator_product(1.0, ['a(q)', 'a*(i)', 'a*(j)', 'a(k)'])
    pq.add_operator_product(1.0, ['a*(l)', 'a(q)'])
    pq.add_operator_product(1.0, ['a(q)', 'a*(l)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()



    print("Generalized-T1 mappings")
    # T1 = {a*(i)a*(j)a*(k), a(n)a(m)a(l)}
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a*(k)', 'a(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a*(k)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()







    odd =  of.MajoranaOperator((('i', 'j', 'k')))
    odd2 = of.MajoranaOperator((('i', 'j', 'k')))
    print(odd * odd2 + odd2 * odd)
    exit()

    # i_term = of.MajoranaOperator((('i'))) - 1j *  of.MajoranaOperator((('I')))
    # j_term = of.MajoranaOperator((('j'))) - 1j *  of.MajoranaOperator((('J')))
    # k_term = of.MajoranaOperator((('k'))) + 1j *  of.MajoranaOperator((('K')))
    # n_term = of.MajoranaOperator((('n'))) - 1j *  of.MajoranaOperator((('N')))
    # m_term = of.MajoranaOperator((('l'))) + 1j *  of.MajoranaOperator((('L')))
    # l_term = of.MajoranaOperator((('i'))) + 1j *  of.MajoranaOperator((('I')))
    
    # t2_term_1 = i_term * j_term * k_term * n_term * m_term * l_term    
    # t2_term_2 = n_term * m_term * l_term * i_term * j_term * k_term 
    # print(t2_term_1 + t2_term_2)
    # exit()
    
    
    # i_term = of.MajoranaOperator((('i'))) - 1j *  of.MajoranaOperator((('I')))
    # j_term = of.MajoranaOperator((('j'))) - 1j *  of.MajoranaOperator((('J')))
    # k_term = of.MajoranaOperator((('k'))) - 1j *  of.MajoranaOperator((('K')))
    # n_term = of.MajoranaOperator((('n'))) + 1j *  of.MajoranaOperator((('N')))
    # m_term = of.MajoranaOperator((('m'))) + 1j *  of.MajoranaOperator((('M')))
    # l_term = of.MajoranaOperator((('l'))) + 1j *  of.MajoranaOperator((('L')))
    
    # t1_term_1 = i_term * j_term * k_term * n_term * m_term * l_term    
    # t1_term_2 = n_term * m_term * l_term * i_term * j_term * k_term 
    
    # i_term = of.MajoranaOperator((0,))  - 1j * of.MajoranaOperator((1,))
    # j_term = of.MajoranaOperator((2,))  - 1j * of.MajoranaOperator((3,))
    # k_term = of.MajoranaOperator((4,))  - 1j * of.MajoranaOperator((5,))
    # n_term = of.MajoranaOperator((6,))  + 1j * of.MajoranaOperator((7,))
    # m_term = of.MajoranaOperator((8,))  + 1j * of.MajoranaOperator((9,))
    # l_term = of.MajoranaOperator((10,)) + 1j * of.MajoranaOperator((11,))
    
    # t1_term_1 = i_term * j_term * k_term * n_term * m_term * l_term    
    # t1_term_2 = n_term * m_term * l_term * i_term * j_term * k_term 


    # print("T2 mappings - damaz")
    # pq = pdaggerq.pq_helper('true')
    # pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a(k)', 'a*(n)', 'a(m)', 'a(l)'])
    # pq.add_operator_product(1.0, ['a(i)', 'a(j)', 'a*(k)', 'a(n)', 'a*(m)', 'a*(l)'])
    # pq.simplify()
    # pq.print(string_type = 'all')
    # pq.clear()
    # note how there is a three body term. what we have written is not the anticommutator between 

    # print("Q -> D")
    # pq.add_operator_product(1.0, ['a(i)', 'a(j)', 'a*(k)', 'a*(l)'])
    # pq.simplify()
    # pq.print(string_type = 'all')
    # pq.clear()

    # print("G -> D")
    # pq.add_operator_product(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'])
    # pq.simplify()
    # pq.print(string_type = 'all')
    # pq.clear()

if __name__ == "__main__":
    main()