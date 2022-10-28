import numpy as np
from numpy import linalg
# np.set_printoptions(threshold=np.nan)
import scipy
import ase
# from pyscf import gto
from pyscf.pbc import gto as pbcgto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
# from helpers import get_ase_diamond_primitive, build_cell
import time
import matplotlib
import matplotlib.pyplot as plt
import sys

#Constants
eVtoEh=27.21138602
pi=np.pi

#Input
kpts=[5,5,5] #number of k-points in each dimension
ecut=800./eVtoEh #ecut is the kinetic energy cutoff in Eh
mss=5 #DIIS max subspace size
alphamix=1.0 #alpha is the mixing parameter for the SCF

xalpha=2./3. #value of alpha in the X-alpha method
maxiter=50 #maxiter is the max number of SCF iterations
threshold=1.e-6 #SCF convergence criteria
eps=1.e-8 #thresh for small |G|^2
numdiv=10 #number of divisions between special points

#User Input
if len(sys.argv)==5:
    kpts=eval(sys.argv[1])
    ecut=float(sys.argv[2])/eVtoEh
    mss=int(sys.argv[3])
    alphamix=float(sys.argv[4])

#Lattice/Cell
lc=10.26 #lattice constant
atomtype='Si'
unittype='diamond'

#FIGURE OUT HOW TO GET THIS FROM PYSCF!!!
nelec=8
nbands=4

def Get_Loc_PSP_AH_Elem(g2):
    v1=3.042
    v2=-1.372
    alp=0.6102
    Zion=4.

    if g2>eps:
        vsg=np.exp(-g2/(4.*alp))*(-4.*pi*Zion/g2+(pi/alp)**(1.5)*(v1+(v2/alp)*(1.5-g2/(4.*alp))))
    else: #|G|^2->0 limit
        vsg=pi*Zion/alp+((pi/alp)**(1.5))*(v1+1.5*v2/alp)

    return vsg/omega

def Get_Loc_PSP_AH_Vec(g2):
    v1=3.042
    v2=-1.372
    alp=0.6102
    Zion=4.

    vsg=np.zeros(len(g2),dtype='float64')
    largeind=g2>eps
    smallind=g2<=eps #|G|^2->0 limit
    g2=g2[largeind]
    vsgl=np.exp(-g2/(4.*alp))*(-4.*pi*Zion/g2+(pi/alp)**(1.5)*(v1+(v2/alp)*(1.5-g2/(4.*alp))))
    vsgs=pi*Zion/alp+((pi/alp)**(1.5))*(v1+1.5*v2/alp) #|G|^2->0 limit
    vsg[largeind]=vsgl
    vsg[smallind]=vsgs

    return vsg/omega

def Get_Loc_PSP_GTH_Elem(g2):
    c1=-7.336103
    c2=0.
    c3=0.
    c4=0.
    rloc=0.44
    Zion=4.

    #Uncomment below to run AH through GTH
    #v1=3.042
    #v2=-1.372
    #alp=0.6102
    #rloc=np.sqrt(1./(2.*alp))
    #c1=v1
    #c2=v2/(2.*alp)

    rloc2=rloc*rloc
    rloc3=rloc*rloc2
    rloc4=rloc2*rloc2
    rloc6=rloc2*rloc4
    g4=g2*g2
    g6=g2*g4

    if g2>eps:
        vsg=np.exp(-g2*rloc2/2.)*(-4.*pi*Zion/g2+np.sqrt(8.*pi**3.)*rloc3*(c1+c2*(3.-g2*rloc2)+c3*(15.-10.*g2*rloc2+g4*rloc4)+c4*(105.-105.*g2*rloc2+21.*g4*rloc4-g6*rloc6)))
    else: #|G|^2->0 limit
        vsg=2.*pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*pi)*rloc+Zion)

    return vsg/omega

def Get_Loc_PSP_GTH_Vec(g2):
    c1=-7.336103
    c2=0.
    c3=0.
    c4=0.
    rloc=0.44
    Zion=4.

    #Uncomment below to run AH through GTH
    #v1=3.042
    #v2=-1.372
    #alp=0.6102
    #rloc=np.sqrt(1./(2.*alp))
    #c1=v1
    #c2=v2/(2.*alp)

    vsg=np.zeros(len(g2),dtype='float64')
    largeind=g2>eps
    smallind=g2<=eps #|G|^2->0 limit
    g2=g2[largeind]
    rloc2=rloc*rloc
    rloc3=rloc*rloc2
    rloc4=rloc2*rloc2
    rloc6=rloc2*rloc4
    g4=g2*g2
    g6=g2*g4

    vsgl=np.exp(-g2*rloc2/2.)*(-4.*pi*Zion/g2+np.sqrt(8.*pi**3.)*rloc3*(c1+c2*(3.-g2*rloc2)+c3*(15.-10.*g2*rloc2+g4*rloc4)+c4*(105.-105.*g2*rloc2+21.*g4*rloc4-g6*rloc6)))
    vsgs=2.*pi*rloc2*((c1+3.*(c2+5.*(c3+7.*c4)))*np.sqrt(2.*pi)*rloc+Zion) #|G|^2->0 limit
    vsg[largeind]=vsgl
    vsg[smallind]=vsgs

    return vsg/omega

def Get_NonLoc_PSP_GTH_Store(gv):
    #Si only
    #0,1 ang mom only
    rgv,thetagv,phigv=pbcgto.pseudo.pp.cart2polar(gv)
    r0=0.422738
    r1=0.484278
    rl=[r0,r1]

    lmax=2
    mmax=2*(lmax-1)+1
    imax=2
    gmax=len(gv)
    SHstore=np.zeros((lmax,mmax,gmax),dtype='complex128')
    Pstore=np.zeros((lmax,imax,gmax),dtype='complex128')
    for l in range(lmax):
        for m in range(-l,l+1):
            SHstore[l,m+l,:]=scipy.special.sph_harm(m,l,phigv,thetagv)
        for i in range(imax):
            Pstore[l,i,:]=pbcgto.pseudo.pp.projG_li(rgv,l,i,rl[l])

    return SHstore,Pstore

def Get_NonLoc_PSP_GTH_Vec(sphg,pg,gind):
    #Si only
    #0,1 ang mom only
    hgth=np.zeros((2,3,3),dtype='float64')
    hgth[0,0,0]=5.906928
    hgth[0,1,1]=3.258196
    hgth[0,0,1]=hgth[0,1,0]=-0.5*np.sqrt(3./5.)*hgth[0,1,1]
    hgth[1,0,0]=2.727013

    vsg=0.
    for l in [0,1]:
        vsgij=vsgsp=0.
        for i in [0,1]:
            for j in [0,1]:
                #vsgij+=thepow[l]*pg[l,i,gind]*hgth[l,i,j]*pg[l,j,:]
                vsgij+=pg[l,i,gind]*hgth[l,i,j]*pg[l,j,:]
        for m in range(-l,l+1):
            vsgsp+=sphg[l,m+l,gind]*sphg[l,m+l,:].conj()
        vsg+=vsgij*vsgsp

    return vsg/omega

def GetMP():
    ase_atom=ase.build.bulk(atomtype, unittype, a=lc)
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell #lattice vectors
    cell.ke_cutoff=ecut #kinetic energy cutoff
    cell.precision=1.e-8
    cell.dimension = 3 #3D PBC
    cell.unit = 'B' #Bohr
    cell.build()
    k=cell.make_kpts(kpts,wrap_around=True) #get k-points from PySCF
    a=cell.a #real lattice vectors
    h=cell.reciprocal_vectors() #reciprocal lattice vectors
    omega=np.linalg.det(a) #cell volume

    return k,a,h,omega,cell

def GetIBZ():
    specialpoints=(2.*pi/lc)*np.array([[0.5,0.5,0.5],[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.5,0.0],[0.75,0.75,0.0],[0.0,0.0,0.0]])
    xaxis=np.zeros((len(specialpoints)-1)*numdiv+1,dtype='float64')
    k=np.zeros(shape=[1,3],dtype='float64')
    k[0]=specialpoints[0]
    ind=0
    for i in range(len(specialpoints)-1):
        vec=specialpoints[i+1]-specialpoints[i]
        inc=vec/float(numdiv)
        for j in range(numdiv):
            ind+=1
            k=np.concatenate((k,np.expand_dims(specialpoints[i]+(j+1)*inc,axis=0)))
            xaxis[ind]=np.linalg.norm(inc)+xaxis[ind-1]

    return k,xaxis

def GetGrids(cutoff):
    #estimate maximum values for the miller indices (for density)
    nm1=int(np.ceil((np.sqrt(2.*4.*cutoff)/(2.*pi))*np.linalg.norm(a[0])+1.))
    nm2=int(np.ceil((np.sqrt(2.*4.*cutoff)/(2.*pi))*np.linalg.norm(a[1])+1.))
    nm3=int(np.ceil((np.sqrt(2.*4.*cutoff)/(2.*pi))*np.linalg.norm(a[2])+1.))

    #g, g2, and mill
    g=np.empty(shape=[0,3],dtype='float64')
    g2=np.empty(shape=[0,0],dtype='float64')
    mill=np.empty(shape=[0,3],dtype='int')
    for i in np.arange(-nm1,nm1+1):
        for j in np.arange(-nm2,nm2+1):
            for k in np.arange(-nm3,nm3+1):
                gtmp=i*h[0]+j*h[1]+k*h[2] # G vector
                g2tmp=np.dot(gtmp,gtmp) # |G|^2
                if (g2tmp/2.<=4.*cutoff): #cutoff for density is 4 times cutoff for orbitals
                    g=np.concatenate((g,np.expand_dims(gtmp,axis=0))) # collect G vectors
                    g2=np.append(g2,g2tmp) # collect |G|^2
                    mill=np.concatenate((mill,np.expand_dims(np.array([i,j,k]),axis=0))) #list of miller indices for G vectors

    #nm contains the maximum dimension for reciprocal basis
    #nr is the real space grid dimensions
    nm=[int(np.amax(mill[:,0])),int(np.amax(mill[:,1])),int(np.amax(mill[:,2]))]
    nr=[2*nm[0]+1,2*nm[1]+1,2*nm[2]+1]

    #indg contains the index of G vectors by their miller indices
    indg=np.ones(nr,dtype='int')*1000000
    for i in range(len(g)):
        indg[mill[i,0]+nm[0],mill[i,1]+nm[1],mill[i,2]+nm[2]]=i

    #increment the size of the real space grid until it is FFT-ready (only contains factors of 2, 3, or 5)
    for i in range(len(nr)):
        while np.any(np.union1d(FactorInteger(nr[i]),[2,3,5])!=[2,3,5]):
            nr[i]+=1

    return g,g2,mill,nm,nr,indg

def GetK(cutoff,k,g):
    #npw contains len(k) elements, each of which indicates the number of orbital G vectors
    npw=np.zeros(len(k),dtype='int')
    for i in range(len(k)):
        for j in range(len(g)):
            kgtmp=k[i]+g[j]
            kg2tmp=np.dot(kgtmp,kgtmp)
            if(kg2tmp/2.<=cutoff):
                npw[i]+=1

    #indgk contains (per k-point) the index of G vectors used for the orbitals
    indgk=np.ones((len(k),np.amax(npw)),dtype='int')*1000000
    for i in range(len(k)):
        ind=0
        for j in range(len(g)):
            kgtmp=k[i]+g[j]
            kg2tmp=np.dot(kgtmp,kgtmp)
            if(kg2tmp/2.<=cutoff):
                indgk[i,ind]=j
                ind+=1

    return npw,indgk

def FactorInteger(n):
    i=2
    factors=[]
    while i*i<=n:
        if n%i:
            i+=1
        else:
            n//=i
            if i not in factors:
                factors.append(i)
    if n>1:
        if n not in factors:
            factors.append(n)

    return factors

def getmill(inp):
    m1=mill[inp,0]
    m2=mill[inp,1]
    m3=mill[inp,2]
    if m1<0:
        m1=m1+nr[0]
    if m2<0:
        m2=m2+nr[1]
    if m3<0:
        m3=m3+nr[2]

    return m1,m2,m3

k,a,h,omega,mycell=GetMP()
g,g2,mill,nm,nr,indg=GetGrids(ecut)
npw,indgk=GetK(ecut,k,g)
ngm=len(g)
nk=len(k)
#coords=(pbcgto.Cell.gen_uniform_grids(mycell,(np.array(nr)-1.)/2.)).T #EXX

def GetIndg(inp):
    out=np.zeros(len(inp),dtype='int')
    for i in range(len(inp)):
        out[i]=indg[tuple(inp[i])]

    return out

def GetIndgFaster(inp):
    out=np.zeros(len(inp),dtype='int')
    for i in range(len(inp)):
        out[i]=indg[inp[i,0],inp[i,1],inp[i,2]]

    return out

print("Crystal Axes:")
print(a)
print("Crystal Axes (in units of lc):")
print(a/lc)
print("Reciprocal Axes:")
print(h)
print("Reciprocal Axes (in units of 2*pi/lc):")
print(h/(2.*pi/lc))
print("Number of k points for density =",nk)
print("k points (in units of 2*pi/lc):")
print(k/(2.*pi/lc))
print("")
print("nm is max dim for g(dens): ", nm)
print("nr is length of one dim of cubic real-space grid: ", nr)
print("npw is number of plane waves for orbitals per k-point: ", npw)
print("The number of untruncated plane waves for density would be: ", (nm[0]*2+1)*(nm[1]*2+1)*(nm[2]*2+1))
print("ngm is is number of plane waves for density: ", ngm)
print("nk is number of k-points: ", nk)
print("")

#TODO: Extend to multiple atoms and complex
sg=np.zeros(ngm,dtype='complex128')
for i in range(ngm):
    sg[i]=2.*np.cos(np.dot(g[i],np.array([lc,lc,lc])/8.))

#TODO: Is rho=n/V a good approx?
rhoin=np.ones(nr,dtype='float64')*nelec/omega
drho2=1000.
vg=np.zeros(ngm,dtype='float64')
rhog=np.zeros(ngm,dtype='float64')
rhoinstore=np.empty([0]+nr,dtype='float64')
rhooutstore=np.empty([0]+nr,dtype='float64')

totaltime=time.time()
for i in range(maxiter):
    SCFcyc=i+1
    if i>0:
        print("\nSCF iteration took ",time.time()-scfitertime, " seconds.")
    else:
        print("Beginning SCF!")

    scfitertime=time.time()
    rhoout=np.zeros(nr,dtype='float64')

    for j in range(nk):

        Htime=time.time()
        hmat=np.zeros((npw[j],npw[j]),dtype='complex128')
        gkind=indgk[j,:npw[j]]
        gk=g[gkind]
        sphg,pg=Get_NonLoc_PSP_GTH_Store(k[j]+gk)
        for aa in range(npw[j]):
            ik=indgk[j][aa]
            gdiff=mill[ik]-mill[gkind[aa:]]+np.array(nm)
            inds=indg[gdiff.T.tolist()]
            vsg=Get_Loc_PSP_GTH_Vec(g2[inds])+Get_NonLoc_PSP_GTH_Vec(sphg,pg,aa)[aa:]
            hmat[aa,aa:]=vsg*sg[inds]+vg[inds]
        kgtmp=k[j]+g[gkind]
        thediag=np.einsum('ij,ij->i',kgtmp,kgtmp)/2.+hmat.diagonal()
        np.fill_diagonal(hmat,thediag)
        print("Forming Hamiltonian took", time.time()-Htime, "seconds.")

        DIAGtime=time.time()
        eval,psi=scipy.linalg.eigh(hmat,lower=False,eigvals=(0,nbands-1))
        print("k point (in units of 2*pi/lc):", k[j]/(2.*pi/lc))
        print("Band energies (in eV):", eval[:nbands]*eVtoEh)
#        psistore[j,:npw[j],:nbands]=psi[:,:nbands] #EXX
        print("Diagonalization took", time.time()-DIAGtime, "seconds.")

        for pp in range(nbands):
            aux=np.zeros(nr,dtype='complex128')
            for tt in range(npw[j]):
                ik=indgk[j][tt]
                aux[getmill(ik)]=psi[tt,pp]
            aux=(1./np.sqrt(omega))*np.fft.fftn(aux)
            rhoout+=(2./nk)*np.absolute(aux)**2.

    MIXtime=time.time()
    charge=(omega/(nr[0]*nr[1]*nr[2]))*np.sum(np.absolute(rhoout))
    if np.absolute(charge-nelec)>eps*10:
        print("Check: Charge (Real Space): ", charge)

    if SCFcyc<=mss:
        rhoinstore=np.concatenate((rhoinstore,np.expand_dims(rhoin,axis=0)))
        rhooutstore=np.concatenate((rhooutstore,np.expand_dims(rhoout,axis=0)))
    else:
        rhoinstore[:mss-1,:,:,:]=rhoinstore[1:,:,:,:]
        rhoinstore[mss-1,:,:,:]=rhoin
        rhooutstore[:mss-1,:,:,:]=rhooutstore[1:,:,:,:]
        rhooutstore[mss-1,:,:,:]=rhoout

    res=rhoout-rhoin
    drho2=np.sqrt(omega)*np.sqrt(np.mean(res**2))

    DIISmatdim=np.amin([SCFcyc,mss])
    Amat=np.zeros((DIISmatdim,DIISmatdim),dtype='float64')
    for cyc1 in range(DIISmatdim):
        for cyc2 in range(DIISmatdim):
            Amat[cyc1,cyc2]=(omega/(nr[0]*nr[1]*nr[2]))*np.sum(np.abs((rhooutstore[cyc1,:,:,:]-rhoinstore[cyc1,:,:,:])*(rhooutstore[cyc2,:,:,:]-rhoinstore[cyc2,:,:,:])))

    alphamat=np.zeros(DIISmatdim,dtype='float64')
    for cyc in range(DIISmatdim):
        alphamat[cyc]=np.sum(1./Amat[:,cyc])/np.sum(1./Amat)

    rhoinnew=0.
    rhooutnew=0.
    for cyc in range(DIISmatdim):
        rhoinnew+=alphamat[cyc]*rhoinstore[cyc,:,:,:]
        rhooutnew+=alphamat[cyc]*rhooutstore[cyc,:,:,:]
    rhoin=alphamix*rhooutnew+(1-alphamix)*rhoinnew

    print("Charge sum and density mix took", time.time()-MIXtime, "seconds.")

    if drho2<threshold:
        print("Convergence threshold reached: ", drho2)
        break
    else:
        print("Iteration ", i+1, "Delta RHO: ", drho2)

    DFTtime=time.time()
    vr=-1.5*xalpha*(3.*rhoin/pi)**(1./3.)
    aux=np.fft.ifftn(vr)
    for ng in range(ngm):
        vg[ng]=np.real(aux[getmill(ng)])
    print("DFT potential took", time.time()-DFTtime, "seconds.")

    Jtime=time.time()
    aux=np.fft.ifftn(rhoin)
    for ng in range(ngm):
        rhog[ng]=np.real(aux[getmill(ng)])
    largeind=g2>eps
    vg[largeind]+=4.*pi*rhog[largeind]/g2[largeind]
    print("Coulomb potential took", time.time()-Jtime, "seconds.")

print("Converging the density took ", (time.time()-totaltime)/60., " minutes.")

k,xaxis=GetIBZ()
nk=len(k)
npw,indgk=GetK(ecut,k,g)
saveeval=np.zeros((nk,np.amin(npw)),dtype='float64')
print("Number of k points for band structure =",nk)
print("k points (in units of 2*pi/lc):")
print(k/(2.*pi/lc))

for j in range(nk):

    Htime=time.time()
    hmat=np.zeros((npw[j],npw[j]),dtype='complex128')
    gkind=indgk[j,:npw[j]]
    gk=g[gkind]
    sphg,pg=Get_NonLoc_PSP_GTH_Store(k[j]+gk)
    for aa in range(npw[j]):
        ik=indgk[j][aa]
        gdiff=mill[ik]-mill[gkind]+np.array(nm)
        inds=indg[gdiff.T.tolist()]
        vsg=Get_Loc_PSP_GTH_Vec(g2[inds])+Get_NonLoc_PSP_GTH_Vec(sphg,pg,aa)
        hmat[aa,:]=vsg*sg[inds]+vg[inds]
    kgtmp=k[j]+g[gkind]
    thediag=np.einsum('ij,ij->i',kgtmp,kgtmp)/2.+hmat.diagonal()
    np.fill_diagonal(hmat,thediag)
    print("Forming Hamiltonian took", time.time()-Htime, "seconds.")

    DIAGtime=time.time()
    saveeval[j,:]=scipy.linalg.eigvalsh(hmat,eigvals=(0,np.amin(npw)-1))[:np.amin(npw)]
    print("k point (in units of 2*pi/lc):", k[j]/(2.*pi/lc))
    print("Band energies (in eV):", saveeval[j,:nbands]*eVtoEh)
    print("Diagonalization took", time.time()-DIAGtime, "seconds.")

saveeval=(saveeval-np.amax(saveeval[:,:nbands]))*eVtoEh
IBZlist=["L", "$\Gamma$", "X", "W", "K", "$\Gamma$"]
IBG=np.amin(saveeval[:,nbands:])-np.amax(saveeval[:,:nbands])
DBG=np.amin(saveeval[numdiv*IBZlist.index('$\\Gamma$'),nbands:])-np.amax(saveeval[numdiv*IBZlist.index('$\\Gamma$'),:nbands])

print("Band structure (in eV):")
#print saveeval[:,:nbands*5]
print(saveeval[:,:8])

print("The indirect band gap is ", IBG, " eV.")
print("The direct band gap is ", DBG, " eV.")

def PlotBandGap():
    for i in range(np.amin(npw)):
        plt.plot(xaxis,saveeval[:,i],'-o',markersize=5,linewidth=1)
    plt.xlim=([np.amin(xaxis),np.amax(xaxis)])
    plt.ylim([-11,11])
    plt.xticks(xaxis[0::numdiv],IBZlist,fontsize=20);
    plt.yticks(range(-10,15,5),fontsize=20);
    plt.show()

PlotBandGap()

print("Entire calculation took ", (time.time()-totaltime)/60., " minutes.")