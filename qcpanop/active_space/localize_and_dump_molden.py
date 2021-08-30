"""
Read RHF checkpoint get final numbers and get split localized orbitals (doubles, singles, virtuals)

from pyscf import lo
print(mf.mo_occ)
docc_idx = np.where(np.isclose(mf.mo_occ, 2.))[0]
print(docc_idx)
socc_idx = np.where(np.isclose(mf.mo_occ, 1.))[0]
print(socc_idx)
virt_idx = np.where(np.isclose(mf.mo_occ, 0.))[0]
print(virt_idx)

loc_docc_mo = lo.ER(mol, mf.mo_coeff[:, docc_idx], verbose=4).kernel()
loc_socc_mo = lo.ER(mol, mf.mo_coeff[:, socc_idx], verbose=4).kernel()
loc_virt_mo = lo.ER(mol, mf.mo_coeff[:, vocc_idx], verbose=4).kernel()


"""
import numpy as np
from pyscf import gto, scf, lo
from pyscf.tools import molden
from pyscf.mcscf.PiOS import MakePiOS

def main():
    scf_type = 'rohf'
    spin = 5  # 2S not 2S + 1
    basis = 'ccpvdz'
    loc_type = 'noloc'
    chkfile_path = 'heme_cys_rohf_ccpvdz_mult6.chk'

    # scf_dict = {'e_tot', 'mo_coeff', 'mo_occ', 'mo_energy'}
    mol, scf_dict = scf.chkfile.load_scf(chkfile_path)

    docc_idx = np.where(np.isclose(scf_dict['mo_occ'], 2.))[0]
    socc_idx = np.where(np.isclose(scf_dict['mo_occ'], 1.))[0]
    virt_idx = np.where(np.isclose(scf_dict['mo_occ'], 0.))[0]

    if loc_type == 'pm':
        print("Localizing doubly occupied")
        loc_docc_mo = lo.PM(mol, scf_dict['mo_coeff'][:, docc_idx]).kernel(verbose=5)
        print()
        print("Localizing singly occupied")
        loc_socc_mo = lo.PM(mol, scf_dict['mo_coeff'][:, socc_idx]).kernel(verbose=5)
        print()
        print("Localizing virtual")
        loc_virt_mo = lo.PM(mol, scf_dict['mo_coeff'][:, virt_idx]).kernel(verbose=5)

    if loc_type == 'er':
        print("Localizing doubly occupied")
        loc_docc_mo = lo.ER(mol, scf_dict['mo_coeff'][:, docc_idx]).kernel(verbose=5)
        print()
        print("Localizing singly occupied")
        loc_socc_mo = lo.ER(mol, scf_dict['mo_coeff'][:, socc_idx]).kernel(verbose=5)
        print()
        print("Localizing virtual")
        loc_virt_mo = lo.ER(mol, scf_dict['mo_coeff'][:, virt_idx]).kernel(verbose=5)

    if loc_type == 'noloc':
        loc_docc_mo = scf_dict['mo_coeff'][:, docc_idx]
        loc_socc_mo = scf_dict['mo_coeff'][:, socc_idx]
        loc_virt_mo = scf_dict['mo_coeff'][:, virt_idx]

    if loc_type == 'ibo':
        print("Localizing doubly occupied")
        loc_docc_mo = lo.ibo.ibo(mol, scf_dict['mo_coeff'][:, docc_idx], locmethod='IBO')
        print()
        print("Localizing singly occupied")
        loc_socc_mo = lo.ibo.ibo(mol, scf_dict['mo_coeff'][:, socc_idx], locmethod='IBO')
        print()
        print("Localizing virtual")
        loc_virt_mo = lo.ibo.ibo(mol, scf_dict['mo_coeff'][:, virt_idx], locmethod='IBO')

    if loc_type == 'boys':
        print("Localizing doubly occupied")
        loc_docc_mo = lo.Boys(mol, scf_dict['mo_coeff'][:, docc_idx]).kernel(verbose=5)
        print()
        print("Localizing singly occupied")
        loc_socc_mo = lo.Boys(mol, scf_dict['mo_coeff'][:, socc_idx]).kernel(verbose=5)
        print()
        print("Localizing virtual")
        loc_virt_mo = lo.Boys(mol, scf_dict['mo_coeff'][:, virt_idx]).kernel(verbose=5)

    loc_mo_coeff = np.hstack((loc_docc_mo, loc_socc_mo, loc_virt_mo))
    np.save("{}_localized_{}_{}_spin{}_localized_mocoeffs".format(loc_type, scf_type, basis, spin), loc_mo_coeff)


    localized_chkfile_name = '{}_localized_{}_{}_spin{}.chk'.format(loc_type, scf_type, basis, spin)
    scf.chkfile.dump_scf(mol, localized_chkfile_name, scf_dict['e_tot'], scf_dict['mo_energy'], loc_mo_coeff, scf_dict['mo_occ'])

    molden_filename = '{}_localized_{}_{}_spin{}.molden'.format(loc_type, scf_type, basis, spin)
    molden.from_chkfile(molden_filename, localized_chkfile_name)

if __name__ == "__main__":
    main()
