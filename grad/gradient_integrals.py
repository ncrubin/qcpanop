"""
AO gradient tensors (core, eri, s) for MO format of the gradient
"""
import numpy as np
from pyscf import gto


def hcore_generator(mol: gto.Mole):
    """Generator for the core deriv function

    int1e_ipkin and int1e_ipnuc take the grad with respect to each
    basis function's atomic position x, y, z and place in a matrix.
    To get the gradient with respect to a particular atom we must
    add the columns of basis functions associated
    """
    aoslices = mol.aoslice_by_atom()
    h1 = mol.intor('int1e_ipkin', comp=3)  #(0.5 \nabla | p dot p | \)
    h1 += mol.intor('int1e_ipnuc', comp=3)  #(\nabla | nuc | \)
    h1 *= -1  # what is this for?

    def hcore_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        # this part gets the derivative with respect to the electron-nuc
        # operator. See pyscf docs for more info. (p|Grad_{Ra}Sum(M) 1/r_{e} - R_{M}|q)
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += h1[:, p0:p1]  # add the row's that aren't zero
        return vrinv + vrinv.transpose(0, 2, 1)

    return hcore_deriv


def overlap_generator(mol: gto.Mole):
    aoslices = mol.aoslice_by_atom()
    s1 = mol.intor('int1e_ipovlp', comp=3)  # (.5 nabla \| p dot p\)

    def ovlp_deriv(atm_id):
        s_r = np.zeros_like(s1)
        shl0, shl1, p0, p1 = aoslices[atm_id]
        # row-idx indexes basis function.  All basis functions not on
        # a specific atom is zero.
        s_r[:, p0:p1] = s1[:, p0:p1]
        # (.5 nabla \| p dot p \ ) +  (\| p dot p| .5 nabla)
        return s_r + s_r.transpose((0, 2, 1))

    return ovlp_deriv


def eri_generator(mol: gto.Mole):
    """Using int2e_ip1 = (nabla, | , )

    Remeber: chem notation (1*,1|2*,2) -> (ij|kl)

    NOTE: Prove the following is true through integral recursions

    (nabla i,j|kl) = (j,nablai|k,l) = (k,l|nabla i,j) = (k,l|j,nabla i)
    """
    aoslices = mol.aoslice_by_atom()
    eri_3 = mol.intor("int2e_ip1", comp=3)

    def eri_deriv(atm_id):
        eri_r = np.zeros_like(eri_3)
        shl0, shl1, p0, p1 = aoslices[atm_id]
        # take only the p0:p1 rows of the first index.
        # note we leverage numpy taking over all remaining jkl indices.
        # (p1 - p0, N, N, N) are non-zero
        eri_r[:, p0:p1] = eri_3[:, p0:p1]
        eri_r[:, :, p0:p1, :, :] += np.einsum('xijkl->xjikl', eri_3[:, p0:p1])
        eri_r[:, :, :, p0:p1, :] += np.einsum('xijkl->xklij', eri_3[:, p0:p1])
        eri_r[:, :, :, :, p0:p1] += np.einsum('xijkl->xklji', eri_3[:, p0:p1])
        return eri_r

    return eri_deriv


def grad_nuc(mol: gto.Mole, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates

    courtesy of pyscf and Szabo
    '''
    gs = np.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.sqrt(np.dot(r1-r2,r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs



if __name__ == "__main__":
    np.set_printoptions(linewidth=500)
    mol = gto.M(
        verbose=0,
        atom='Li 0 0 0; H 0 0 1.5',
        basis='sto-3g',
    )


    aoslices = mol.aoslice_by_atom()
    print(aoslices)
    atm_id = 0
    shl0, shl1, p0, p1 = aoslices[atm_id]

    eri_3 = mol.intor("int2e_ip1", comp=3)
    print(eri_3.shape)

