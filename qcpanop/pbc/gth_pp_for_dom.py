import numpy as np
import warnings
from pyscf.data.elements import ELEMENTS
from pyscf.pbc import gto
from pyscf.lib.exceptions import BasisNotFoundError


def print_l_hij(atom_symb):
    cell = gto.M(a=np.eye(3) * 10,
                 atom='{} 0 0 0'.format(atom_symb),
                 basis='sto-3g',
                 pseudo='gth-blyp',
                 unit='angstrom',
                 ke_cutoff=0.5, )
    cell.exp_to_discard = 0.1
    cell.build()

    pp = cell._pseudo[atom_symb]
    for l, proj in enumerate(pp[5:]):
        # print("l ", l, " proj ", proj)
        rl, nl, hl = proj
        hl = np.asarray(hl)
        h_mat_to_print = np.zeros((3, 3), dtype=hl.dtype)
        if nl > 0:
            h_mat_to_print[:hl.shape[0], :hl.shape[1]] = hl
            print('l = {}'.format(l))
            print(h_mat_to_print)


def main():
    elements_with_gth_blyp = []
    for element in ELEMENTS:
        try:
            cell = gto.M(a=np.eye(3) * 10,
                             atom='{} 0 0 0'.format(element),
                             basis='sto-3g',
                             pseudo='gth-blyp',
                             unit='angstrom',
                             ke_cutoff=0.5,)
            cell.exp_to_discard = 0.1
            print(cell.nelec)
            cell.build()
        except (BasisNotFoundError, RuntimeError):
            continue
        print('element {} has gth pp '.format(element))
        elements_with_gth_blyp.append(element)


    print("elements that have GTH-BLYP PP")
    print(elements_with_gth_blyp)
    for ee in elements_with_gth_blyp:
        print(ee)
        print_l_hij(ee)

    print("COBALT OUTPUT")
    print_l_hij('Co')



main()