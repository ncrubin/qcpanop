"""
Implement gradients
"""
import numpy as np
from rhf import RHF
import ctypes
from pyscf import gto
from pyscf.scf import _vhf


def gradient(rhf_obj: RHF, pyscf_mol):
    """
    Generate the gradient of an RHF object.

    :param rhf_obj: RHF Object
    :param pyscf_mol: pyscf Mole object for nabla integrals
    :return: array((len(atoms), 3)) for X, Y, Z gradient coordinates
    """
    hcore_deriv = hcore_generator(pyscf_mol)
    s1 = -pyscf_mol.intor('int1e_ipovlp', comp=3)
    dm0 = rhf_obj.dmat
    vhf = get_veff(pyscf_mol, dm0)
    atmlst = range(pyscf_mol.natm)
    occ = np.array([2] * rhf_obj.nocc + [0] * (rhf_obj.hcore.shape[0] - rhf_obj.nocc))
    dme0 = make_rdm1e(rhf_obj.mo_energies, rhf_obj.mo_coeff, occ)
    de = np.zeros((len(atmlst), 3))
    aoslices = pyscf_mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xij,ij->x', h1ao, dm0)
        # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[k] += np.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2
        de[k] -= np.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2

    de += grad_nuc(pyscf_mol, atmlst=atmlst)
    return de


def get_jk(mol, dm):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    vhfopt = _vhf.VHFOpt(mol, 'int2e_ip1ip2', 'CVHFgrad_jk_prescreen',
                         'CVHFgrad_jk_direct_scf')
    dm = np.asarray(dm, order='C')
    if dm.ndim == 3:
        n_dm = dm.shape[0]
    else:
        n_dm = 1
    ao_loc = mol.ao_loc_nr()
    fsetdm = getattr(_vhf.libcvhf, 'CVHFgrad_jk_direct_scf_dm')
    fsetdm(vhfopt._this,
           dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
           ao_loc.ctypes.data_as(ctypes.c_void_p),
           mol._atm.ctypes.data_as(ctypes.c_void_p), mol.natm,
           mol._bas.ctypes.data_as(ctypes.c_void_p), mol.nbas,
           mol._env.ctypes.data_as(ctypes.c_void_p))

    # Update the vhfopt's attributes intor.  Function direct_mapdm needs
    # vhfopt._intor and vhfopt._cintopt to compute J/K.  intor was initialized
    # as int2e_ip1ip2. It should be int2e_ip1
    vhfopt._intor = intor = mol._add_suffix('int2e_ip1')
    vhfopt._cintopt = None

    vj, vk = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                               's2kl', # ip1_sph has k>=l,
                               ('lk->s1ij', 'jk->s1il'),
                               dm, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env, vhfopt=vhfopt)
    return -vj, -vk


def get_veff(mol, dm):
    '''NR Hartree-Fock Coulomb repulsion'''
    vj, vk = get_jk(mol, dm)
    return vj - vk * .5


def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:,mo_occ>0]
    mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
    return np.dot(mo0e, mo0.T.conj())


def hcore_generator(mol: gto.Mole):
    aoslices = mol.aoslice_by_atom()
    h1 = mol.intor('int1e_ipkin', comp=3)
    h1 += mol.intor('int1e_ipnuc', comp=3)
    h1 *= -1

    def hcore_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)  # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += h1[:, p0:p1]
        return vrinv + vrinv.transpose(0, 2, 1)

    return hcore_deriv


def grad_nuc(mol, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
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
    np.set_printoptions(linewidth=300)
    from pyscf import gto, scf, grad
    import openfermion as of

    # mol = gto.M(
    #     verbose=0,
    #     atom='O   0.000000000000  -0.143225816552   0.000000000000;H  1.638036840407   1.136548822547  -0.000000000000; H  -1.638036840407   1.136548822547  -0.000000000000',
    #     basis='6-31g',
    # )

    mol = gto.M(
        verbose=0,
        atom='Li 0 0 0; H 0 0 1.5',
        basis='sto-3g',
    )

    mf = scf.RHF(mol)
    mf.kernel()

    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')  # (ij|kl)
    antieri = eri - 0.5 * np.einsum('ijkl->ilkj', eri)  # (ij|kl) - 1/2 (il|kj)
    rhf = RHF(t + v, s, eri, mol.nelectron, iter_max=300,
              diis_length=4)
    rhf.solve_diis()

    dmat = rhf.dmat
    energy = np.einsum('ij,ij', dmat, rhf.hcore) + 0.5 * np.einsum('ij,kl,ijkl',dmat,dmat,antieri)
    print(energy + mol.energy_nuc())

    print(mf.energy_tot())

    g = mf.Gradients()
    print(g.kernel())

    from pyscf.grad.rhf import grad_elec, get_ovlp
    print()
    print(gradient(rhf, mol))



