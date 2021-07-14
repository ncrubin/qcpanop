import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count())
import numpy as np
from numpy import einsum
from openfermion.chem.molecular_data import spinorb_from_spatial


def ccsd_energy(t1, t2, h, g, o, v):
    energy = 0
    #	  1.0000 h(i,i)
    energy += 1.0 * einsum('ii', h[o, o])

    #	  1.0000 h(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', h[o, v], t1)

    #	  0.5000 <i,j||i,j>
    energy += 0.5 * einsum('ijij', g[o, o, o, o])

    #	  1.0000 <i,j||a,j>*t1(a,i)
    energy += 1.0 * einsum('ijaj,ai', g[o, o, v, o], t1)

    #	  0.2500 <i,j||a,b>*t2(a,b,i,j)
    energy += 0.25 * einsum('ijab,abij', g[o, o, v, v], t2)

    #	  0.5000 <i,j||a,b>*t1(a,i)*t1(b,j)
    energy += 0.5 * einsum('ijab,ai,bj', g[o, o, v, v], t1, t1,
                           optimize=['einsum_path', (0, 1), (0, 1)])
    return energy


def singles_residual_contractions(t1, t2, h, g, o, v):
    """
    Singles residual equation with diagonal fock operator terms removed.
    One should use this to generate the residuals and then divide by the fock
    operator energies.

    This residual is implemented in the O(n^5) fashion through optimal
    einsum contractions.  Without finding the optimal contraction ordering
    the einsum contractions would be O(n^6)!

    :param t1: t1 amplitudes ordered t1[v, o]
    :param t2: t2 amplituddes ordered t2[v1, v2, o1, o2]
    :param h: one-electron integrals in spin-orbital basis
    :param g:  antisymmeterized two-electron integrals. not including 1/4 ordered
               in physics notation <1'2'||12>
    :param o: occupied index slice o = slice(None, nocc)
    :param v:  virtual index slice v = slice(nocc, None)
    :return: t1 residuals
    """
    singles_residual = np.zeros_like(t1)
    #	  1.0000 h(e,m)
    singles_residual += 1.0 * einsum('em->em', h[v, o])

    #	  1.0000 h(i,a)*t2(a,e,i,m)
    singles_residual += 1.0 * einsum('ia,aeim->em', h[o, v], t2)

    #	 -1.0000 h(i,a)*t1(a,m)*t1(e,i)
    singles_residual += -1.0 * einsum('ia,am,ei->em', h[o, v], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||i,m>
    singles_residual += 1.0 * einsum('ieim->em', g[o, v, o, o])

    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_residual += 1.0 * einsum('ieam,ai->em', g[o, v, v, o], t1)

    #	  1.0000 <i,j||i,a>*t2(a,e,j,m)
    singles_residual += 1.0 * einsum('ijia,aejm->em', g[o, o, o, v], t2)

    #	 -0.5000 <i,j||a,m>*t2(a,e,i,j)
    singles_residual += -0.5 * einsum('ijam,aeij->em', g[o, o, v, o], t2)

    #	  0.5000 <i,e||a,b>*t2(a,b,i,m)
    singles_residual += 0.5 * einsum('ieab,abim->em', g[o, v, v, v], t2)

    #	 -1.0000 <i,j||i,a>*t1(a,m)*t1(e,j)
    singles_residual += -1.0 * einsum('ijia,am,ej->em', g[o, o, o, v], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t1(e,j)
    singles_residual += -1.0 * einsum('ijam,ai,ej->em', g[o, o, v, o], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_residual += 1.0 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t2(b,e,j,m)
    singles_residual += 1.0 * einsum('ijab,ai,bejm->em', g[o, o, v, v], t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(a,m)*t2(b,e,i,j)
    singles_residual += 0.5 * einsum('ijab,am,beij->em', g[o, o, v, v], t1, t2,
                                     optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t2(a,b,j,m)
    singles_residual += 0.5 * einsum('ijab,ei,abjm->em', g[o, o, v, v], t1, t2,
                                     optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_residual += -1.0 * einsum('ijab,ai,bm,ej->em', g[o, o, v, v], t1,
                                      t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    return singles_residual


def doubles_residual_contractions(t1, t2, h, g, o, v):
    """
    Doubles residual equation with diagonal fock operator terms removed.
    One should use this to generate the residuals and then divide by the fock
    operator energies.

    This residual is implemented in the O(n^6) fashion through optimal
    einsum contractions.  Without finding the optimal contraction ordering
    the einsum contractions would be O(n^8)!

    :param t1: t1 amplitudes ordered t1[v, o]
    :param t2: t2 amplituddes ordered t2[v1, v2, o1, o2]
    :param h: one-electron integrals in spin-orbital basis
    :param g:  antisymmeterized two-electron integrals. not including 1/4
    :param o: occupied index slice o = slice(None, nocc)
    :param v:  virtual index slice v = slice(nocc, None)
    :return: t2 residuals
    """
    doubles_residual = np.zeros_like(t2)
    #	 -1.0000 h(i,a)*t1(a,n)*t2(e,f,m,i)
    doubles_residual += -1.0 * einsum('ia,an,efmi->efmn', h[o, v], t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 h(i,a)*t1(a,m)*t2(e,f,n,i)
    doubles_residual += 1.0 * einsum('ia,am,efni->efmn', h[o, v], t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 h(i,a)*t1(e,i)*t2(a,f,m,n)
    doubles_residual += -1.0 * einsum('ia,ei,afmn->efmn', h[o, v], t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 h(i,a)*t1(f,i)*t2(a,e,m,n)
    doubles_residual += 1.0 * einsum('ia,fi,aemn->efmn', h[o, v], t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <e,f||m,n>
    doubles_residual += 1.0 * einsum('efmn->efmn', g[v, v, o, o])

    #	  1.0000 <i,e||m,n>*t1(f,i)
    doubles_residual += 1.0 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)

    #	 -1.0000 <i,f||m,n>*t1(e,i)
    doubles_residual += -1.0 * einsum('ifmn,ei->efmn', g[o, v, o, o], t1)

    #	  1.0000 <e,f||a,n>*t1(a,m)
    doubles_residual += 1.0 * einsum('efan,am->efmn', g[v, v, v, o], t1)

    #	 -1.0000 <e,f||a,m>*t1(a,n)
    doubles_residual += -1.0 * einsum('efam,an->efmn', g[v, v, v, o], t1)

    #	  0.5000 <i,j||m,n>*t2(e,f,i,j)
    doubles_residual += 0.5 * einsum('ijmn,efij->efmn', g[o, o, o, o], t2)

    #	 -1.0000 <i,e||a,n>*t2(a,f,i,m)
    doubles_residual += -1.0 * einsum('iean,afim->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,e||a,m>*t2(a,f,i,n)
    doubles_residual += 1.0 * einsum('ieam,afin->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,f||a,n>*t2(a,e,i,m)
    doubles_residual += 1.0 * einsum('ifan,aeim->efmn', g[o, v, v, o], t2)

    #	 -1.0000 <i,f||a,m>*t2(a,e,i,n)
    doubles_residual += -1.0 * einsum('ifam,aein->efmn', g[o, v, v, o], t2)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_residual += 0.5 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)

    #	  1.0000 <i,j||m,n>*t1(e,i)*t1(f,j)
    doubles_residual += 1.0 * einsum('ijmn,ei,fj->efmn', g[o, o, o, o], t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,n>*t1(a,m)*t1(f,i)
    doubles_residual += 1.0 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,m>*t1(a,n)*t1(f,i)
    doubles_residual += -1.0 * einsum('ieam,an,fi->efmn', g[o, v, v, o], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,n>*t1(a,m)*t1(e,i)
    doubles_residual += -1.0 * einsum('ifan,am,ei->efmn', g[o, v, v, o], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,m>*t1(a,n)*t1(e,i)
    doubles_residual += 1.0 * einsum('ifam,an,ei->efmn', g[o, v, v, o], t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_residual += -1.0 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||i,a>*t1(a,n)*t2(e,f,m,j)
    doubles_residual += -1.0 * einsum('ijia,an,efmj->efmn', g[o, o, o, v], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||i,a>*t1(a,m)*t2(e,f,n,j)
    doubles_residual += 1.0 * einsum('ijia,am,efnj->efmn', g[o, o, o, v], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,j>*t1(e,i)*t2(a,f,m,n)
    doubles_residual += -1.0 * einsum('ijaj,ei,afmn->efmn', g[o, o, v, o], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,j>*t1(f,i)*t2(a,e,m,n)
    doubles_residual += 1.0 * einsum('ijaj,fi,aemn->efmn', g[o, o, v, o], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(a,i)*t2(e,f,j,m)
    doubles_residual += 1.0 * einsum('ijan,ai,efjm->efmn', g[o, o, v, o], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,n>*t1(a,m)*t2(e,f,i,j)
    doubles_residual += 0.5 * einsum('ijan,am,efij->efmn', g[o, o, v, o], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,n>*t1(e,i)*t2(a,f,j,m)
    doubles_residual += -1.0 * einsum('ijan,ei,afjm->efmn', g[o, o, v, o], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(f,i)*t2(a,e,j,m)
    doubles_residual += 1.0 * einsum('ijan,fi,aejm->efmn', g[o, o, v, o], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t2(e,f,j,n)
    doubles_residual += -1.0 * einsum('ijam,ai,efjn->efmn', g[o, o, v, o], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,m>*t1(a,n)*t2(e,f,i,j)
    doubles_residual += -0.5 * einsum('ijam,an,efij->efmn', g[o, o, v, o], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,m>*t1(e,i)*t2(a,f,j,n)
    doubles_residual += 1.0 * einsum('ijam,ei,afjn->efmn', g[o, o, v, o], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(f,i)*t2(a,e,j,n)
    doubles_residual += -1.0 * einsum('ijam,fi,aejn->efmn', g[o, o, v, o], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    doubles_residual += 1.0 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    doubles_residual += -1.0 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,m)*t2(b,f,n,i)
    doubles_residual += 1.0 * einsum('ieab,am,bfni->efmn', g[o, v, v, v], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    doubles_residual += 0.5 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,i)*t2(b,e,m,n)
    doubles_residual += -1.0 * einsum('ifab,ai,bemn->efmn', g[o, v, v, v], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t2(b,e,m,i)
    doubles_residual += 1.0 * einsum('ifab,an,bemi->efmn', g[o, v, v, v], t1,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,m)*t2(b,e,n,i)
    doubles_residual += -1.0 * einsum('ifab,am,beni->efmn', g[o, v, v, v], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,f||a,b>*t1(e,i)*t2(a,b,m,n)
    doubles_residual += -0.5 * einsum('ifab,ei,abmn->efmn', g[o, v, v, v], t1,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,b,n,j)*t2(e,f,m,i)
    doubles_residual += -0.5 * einsum('ijab,abnj,efmi->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,b,m,j)*t2(e,f,n,i)
    doubles_residual += 0.5 * einsum('ijab,abmj,efni->efmn', g[o, o, v, v], t2,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <i,j||a,b>*t2(a,b,m,n)*t2(e,f,i,j)
    doubles_residual += 0.25 * einsum('ijab,abmn,efij->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,e,i,j)*t2(b,f,m,n)
    doubles_residual += -0.5 * einsum('ijab,aeij,bfmn->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t2(a,e,n,j)*t2(b,f,m,i)
    doubles_residual += 1.0 * einsum('ijab,aenj,bfmi->efmn', g[o, o, v, v], t2,
                                     t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t2(a,e,m,j)*t2(b,f,n,i)
    doubles_residual += -1.0 * einsum('ijab,aemj,bfni->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,e,m,n)*t2(b,f,i,j)
    doubles_residual += -0.5 * einsum('ijab,aemn,bfij->efmn', g[o, o, v, v], t2,
                                      t2,
                                      optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,n>*t1(a,m)*t1(e,j)*t1(f,i)
    doubles_residual += -1.0 * einsum('ijan,am,ej,fi->efmn', g[o, o, v, o], t1,
                                      t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,j||a,m>*t1(a,n)*t1(e,j)*t1(f,i)
    doubles_residual += 1.0 * einsum('ijam,an,ej,fi->efmn', g[o, o, v, o], t1,
                                     t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    doubles_residual += -1.0 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1,
                                      t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t1(b,m)*t1(e,i)
    doubles_residual += 1.0 * einsum('ifab,an,bm,ei->efmn', g[o, v, v, v], t1,
                                     t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    doubles_residual += -1.0 * einsum('ijab,ai,bn,efmj->efmn', g[o, o, v, v],
                                      t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t2(e,f,n,j)
    doubles_residual += 1.0 * einsum('ijab,ai,bm,efnj->efmn', g[o, o, v, v], t1,
                                     t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    doubles_residual += -1.0 * einsum('ijab,ai,ej,bfmn->efmn', g[o, o, v, v],
                                      t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(f,j)*t2(b,e,m,n)
    doubles_residual += 1.0 * einsum('ijab,ai,fj,bemn->efmn', g[o, o, v, v], t1,
                                     t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -0.5000 <i,j||a,b>*t1(a,n)*t1(b,m)*t2(e,f,i,j)
    doubles_residual += -0.5 * einsum('ijab,an,bm,efij->efmn', g[o, o, v, v],
                                      t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,n)*t1(e,j)*t2(b,f,m,i)
    doubles_residual += 1.0 * einsum('ijab,an,ej,bfmi->efmn', g[o, o, v, v], t1,
                                     t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(f,j)*t2(b,e,m,i)
    doubles_residual += -1.0 * einsum('ijab,an,fj,bemi->efmn', g[o, o, v, v],
                                      t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,m)*t1(e,j)*t2(b,f,n,i)
    doubles_residual += -1.0 * einsum('ijab,am,ej,bfni->efmn', g[o, o, v, v],
                                      t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,m)*t1(f,j)*t2(b,e,n,i)
    doubles_residual += 1.0 * einsum('ijab,am,fj,beni->efmn', g[o, o, v, v], t1,
                                     t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_residual += 0.5 * einsum('ijab,ei,fj,abmn->efmn', g[o, o, v, v], t1,
                                     t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_residual += -1.0 * einsum('ijab,an,bm,ei,fj->efmn', g[o, o, v, v],
                                      t1, t1, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 3),
                                                (0, 2), (0, 1)])
    return doubles_residual


def kernel(t1, t2, h, g, o, v, e_ai, e_abij, max_iter=100, stopping_eps=1.0E-8):

    old_energy = ccsd_energy(t1, t2, h, g, o, v)
    for idx in range(max_iter):

        singles_res = singles_residual_contractions(t1, t2, h, g, o, v)
        doubles_res = doubles_residual_contractions(t1, t2, h, g, o, v)

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        current_energy = ccsd_energy(new_singles, new_doubles, h, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx,
                                                                    old_energy,
                                                                    delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles


def prepare_integrals_from_openfermion(molecule):
    oei, tei = molecule.get_integrals()
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    gtei = astei.transpose(0, 1, 3, 2)
    nocc = molecule.n_electrons

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, nocc)
    v = slice(nocc, None)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    h = soei
    g = gtei
    return h, g, o, v, n, e_ai, e_abij, hf_energy


def main():
    from itertools import product
    import pyscf
    import openfermion as of
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np
    from scipy.linalg import expm

    import fqe
    from fqe.openfermion_utils import molecular_data_to_restricted_fqe_op

    from openfermion.chem.molecular_data import spinorb_from_spatial

    basis = 'cc-pvdz'
    mol = pyscf.M(
        atom='H 0 0 0; B 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 1.6)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_ccsd=True)
    hamiltonian = molecule.get_molecular_hamiltonian()
    elec_ham = molecular_data_to_restricted_fqe_op(molecule)
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    nso = 2 * norbs
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    nvirt = norbs - nocc
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    pyscf_astei = np.einsum('ijlk', stei)
    pyscf_astei = pyscf_astei - np.einsum('ijlk', pyscf_astei)


    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    t1s = spatial2spin(mycc.t1)
    t2s = spatial2spin(mycc.t2)

    cc_energy = np.einsum('ia,ia', fock[o, v], t1s) + 0.25 * np.einsum('ijba,ijab', astei[o, o, v, v], t2s) + 0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s)
    print("{: 5.10f}\tNick's CC energy ".format(cc_energy))
    print("{: 5.10f}\tpyscf CC energy".format(molecule.ccsd_energy - molecule.hf_energy))
    print(molecule.ccsd_energy, hf_energy + cc_energy + molecule.nuclear_repulsion)

    # print(hf_energy, np.einsum('ii', soei[o, o]) + 0.5 * np.einsum('ijji', astei[o, o, o, o]))
    # print(np.einsum('ia,ia', fock[o, v], t1s), np.einsum('ia,ia', soei[o, v], t1s) + np.einsum('ijja,ia', astei[o, o, o, v], t1s))
    # print(0.25 * np.einsum('ijba,ijab', astei[o, o, v, v], t2s), 0.25 * np.einsum('ijba,ijab', astei[o,o,v,v], t2s))
    # print(0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s), 0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s))
    print("Check total energy ", ccsd_energy(t1s.transpose(1, 0), t2s.transpose(3, 2, 0, 1), soei, gtei, o, v) - molecule.hf_energy + molecule.nuclear_repulsion)

    h = soei
    g = gtei
    t1 = t1s.transpose(1, 0)
    t2 = t2s.transpose(2, 3, 0, 1)

    #	  1.0000 h(i,i) + 0.5000 <i,j||i,j>
    print(1.0 * einsum('ii', h[o, o]) + 0.5 * einsum('ijij', g[o, o, o, o]), hf_energy, np.einsum('ii', soei[o, o]) + 0.5 * np.einsum('ijji', astei[o, o, o, o]))

    #	  1.0000 h(i,a)*t1(a,i) + 1.0000 <i,j||a,j>*t1(a,i)
    print(1.0 * einsum('ia,ai', h[o, v], t1) + 1.0 * einsum('ijaj,ai', g[o, o, v, o], t1), np.einsum('ia,ia', fock[o, v], t1s))

    #	  0.2500 <i,j||a,b>*t2(a,b,i,j)
    print(0.25 * einsum('ijab,abij', g[o, o, v, v], t2), 0.25 * np.einsum('ijba,ijab', astei[o, o, v, v], t2s))

    #	  0.5000 <i,j||a,b>*t1(a,i)*t1(b,j)
    print(0.5 * einsum('ijab,ai,bj', g[o, o, v, v], t1, t1,
                           optimize=['einsum_path', (0, 1), (0, 1)]),
          0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s), 0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s))

    assert np.allclose(t1 / e_ai, singles_residual_contractions(t1, t2, h, g, o, v))
    assert np.allclose(t2 / e_abij, doubles_residual_contractions(t1,t2, h, g, o ,v))

    t1f, t2f = kernel(np.zeros_like(t1), np.zeros_like(t2), h, g, o, v, e_ai, e_abij)

    h, g, o, v, n, e_ai, e_abij, hf_energy = prepare_integrals_from_openfermion(molecule)
    t1f, t2f = kernel(np.zeros_like(t1), np.zeros_like(t2), h, g, o, v, e_ai,
                      e_abij)
    # cc = CCSD(oei=h, tei=g, nalpha=nocc, nbeta=nocc, escf=hf_energy, orb_energies=np.diagonal(fock))
    # cc.solve_for_amplitudes()

if __name__ == "__main__":
    main()