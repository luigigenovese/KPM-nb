#This modyle contains the method needed for the Lanczos procedure 
import numpy as np

def sp(a,b):
    """
    Perform the scalar product of the row vectors a and b (possibly complex valued)    
    """
    out = np.trace(np.matmul(a.H,b))
    return out

def norm(u):
    """
    Compute the norm of a row vector (possibly complex valued)    
    """
    out = np.sqrt(sp(u,u))
    return out

def nonHermitian_eigensystem(H):
    er,vr = np.linalg.eig(H)
    es,vl = np.linalg.eig(H.H)
    es = es.conj()

    ind = er.argsort()[::-1] 
    er = er[ind]
    vr = vr[:,ind]
    ind = es.argsort()[::-1] 
    es = es[ind]
    vl = vl[:,ind]
    return er,vr,vl

def validate_eigensystem(H,e,w):
    """
    Inspect the validity of the eigensystem e,w as eigenvalues and 
    eigenvectors of the matrix H    
    """
    for ind,e in enumerate(e):
        isok = np.allclose(H.dot(w[:,ind]),e*w[:,ind],atol=1.e-12)
        if not isok: print('Eigenvalue',ind,'is false')
    print('Validation terminated')

def resolvent(u0,e,w,omega,eta=1.e-2):
    G = 0.0
    for ind,E in enumerate(e):
        v=w[:,ind]
        u0dotv = sp(u0,v)
        vdotu0 = sp(v,u0)
        G+=(u0dotv*vdotu0)/(omega-E+1j*eta)
    G = -1/np.pi*G.imag
    return G

def resolvent_nonHermitian(u0,e,wr,wl,omega,eta=1.e-2):
    G = 0.0 
    for ind,E in enumerate(e):
        u0dotphin = sp(u0,wr[:,ind])
        chindotu0 = sp(wl[:,ind],u0)
        G+=u0dotphin*chindotu0/sp(wl[:,ind],wr[:,ind])*1.0/(omega-E+1j*eta)
    G = -1/np.pi*G.imag
    return G

def lanc_recursion(H,phij,phijm1,alphaj,betaj):
    phihatjp1 = H.dot(phij)-alphaj*phij-betaj*phijm1
    betajp1 = norm(phihatjp1)
    phijp1 = phihatjp1/betajp1
    alphajp1 = sp(phijp1,H.dot(phijp1))
    return phijp1,alphajp1,betajp1

def lanc_recursion_nonHermitian(H,phij,phijm1,chij,chijm1,alphaj,betaj,gammaj):
    phihatjp1 = H.dot(phij)-alphaj*phij-gammaj*phijm1
    chihatjp1 = chij.dot(H)-chij*alphaj-chijm1*betaj
    betajp1 = norm(phihatjp1)
    gammajp1 = sp(chihatjp1.H,phihatjp1)/betajp1
    phijp1 = phihatjp1/betajp1
    chijp1=chihatjp1/gammajp1
    alphajp1 = sp(chijp1.H,H.dot(phijp1))
    return phijp1,chijp1,alphajp1,betajp1,gammajp1

def ortCheck(krylovl,krylovr,dimVal):
    """
    Check the orthonormality of the left and right Krylov spaces
    """
    klkr = np.matmul(krylovl,krylovr)
    klkrReduced = klkr[0:dimVal,0:dimVal]
    isok = np.allclose(klkrReduced,np.matrix(np.eye(dimVal)),atol=1.e-10)
    return isok

def Tmatrix(alpha,beta,gamma,dimVal):
    """
    Compute a tridiagonal matrix of dimension (dimVal,dimVal) with elements given by the alpha, beta
    and gamma parameters 
    """
    T = np.matrix(np.diag(alpha),dtype=np.complex_)
    Treduced = T[0:dimVal,0:dimVal]
    Up = np.matrix(np.zeros(dimVal**2,dtype=np.complex_).reshape(dimVal,dimVal))
    Down = np.matrix(np.zeros(dimVal**2,dtype=np.complex_).reshape(dimVal,dimVal))
    for i in range(dimVal-1):
        Up[i,i+1] = gamma[i+1]
        Down[i+1,i] = beta[i+1]
    Treduced += Up + Down
    
    return Treduced
    
def tridiagCheck(H,krylovl,krylovr,alpha,beta,gamma,dimVal):
    """
    Check if the matrix H is tridiagonal, with elements given by the alpha, beta
    and gamma parameters, when expressed in the basis of the Krylov vectors
    """
    Hkrylov = np.matmul(krylovl,H.dot(krylovr))
    HkrylovReduced = Hkrylov[0:dimVal,0:dimVal]
    isok = np.allclose(HkrylovReduced,Tmatrix(alpha,beta,gamma,dimVal),atol=1.e-10)
    return isok

def D(i,H,omega,eta=1.e-2):
    DimH = len(H)
    Hsub = H[i:DimH,i:DimH]
    dimHsub = len(Hsub)
    Di = (omega+1j*eta)*np.eye(dimHsub)-Hsub
    Didet = np.linalg.det(Di)
    return Didet

def Gdet(H,omega,eta=1.e-2):
    D0det = D(0,H,omega,eta)
    D1det = D(1,H,omega,eta)
    ratio = D1det/D0det
    out = -1/np.pi*ratio.imag
    return out

def cont_frac(alpha,beta,gamma,omega,eta=1.e-2):
    alphabetagamma=[(a,b,g) for a,b,g in zip(alpha,beta,gamma)]
    f=0.0j
    for a,b,g in reversed(alphabetagamma):
        f = 1.0/(omega + 1j*eta - a - (b*g)*f)
    return f

def Gfrac(alpha,beta,gamma,omega,eta=1.e-2):
    betacut = beta[1:len(beta)]
    gammacut = gamma[1:len(gamma)]
    frac = cont_frac(alpha,betacut,gammacut,omega,eta)
    out = -1/np.pi*frac.imag
    return out

class KrylovLoop():
    def __init__(self,dimKrylov,H,phi0,norm_=norm,recursion_=lanc_recursion):
        
        # definition and init
        self.dimKrylov=dimKrylov
        self.H=H
        self.alpha=[]
        self.beta=[]
        n,n1=self.H.shape
        assert n == n1
        self.recursion=recursion_
        self.krylov = np.matrix(np.zeros(n*dimKrylov).reshape(n,dimKrylov))
        
        self.krylov[:,0] = phi0
        alpha0 = sp(phi0,H.dot(phi0))
        self.alpha.append(alpha0)
        self.beta.append(0)
        self.dimVal = 1

        # first iteration
        unot = np.matrix(np.zeros(n).reshape(n,1)) #needed for the first step
        self.__iterate(self.krylov[:,0],unot)
        self.dimVal += 1
        
    def __iterate(self,phi_j,phi_jm1):
        phi_jp1,alpha_jp1,beta_jp1=self.recursion(self.H,phi_j,phi_jm1,self.alpha[-1],self.beta[-1])
        self.krylov[:,self.dimVal] = phi_jp1
        self.alpha.append(alpha_jp1)
        self.beta.append(beta_jp1)
        
    def iterate(self,niter=None):
        # generic iteration
        nit=niter if niter is not None else self.dimKrylov
        while self.dimVal < nit :
            self.__iterate(self.krylov[:,self.dimVal-1],self.krylov[:,self.dimVal-2])
            self.dimVal+=1
            print('Dimension Krylov space = ',self.dimVal,'Orthogonality check = ',ortCheck(self.krylov.H,self.krylov,self.dimVal),'Tridiagonal check = ',tridiagCheck(self.H,self.krylov.H,self.krylov,self.alpha,self.beta,self.beta,self.dimVal))

class KrylovLoop_nonHermitian():
    def __init__(self,dimKrylov,H,phi0,norm_=norm,recursion_=lanc_recursion_nonHermitian):
        
        # definition and init
        self.dimKrylov=dimKrylov
        self.H=H
        self.alpha=[]
        self.beta=[]
        self.gamma=[]
        n,n1=self.H.shape
        assert n == n1
        self.recursion=recursion_
        self.krylovr = np.matrix(np.zeros((n*dimKrylov),dtype=np.complex_).reshape(n,dimKrylov))
        self.krylovl = np.matrix(np.zeros((n*dimKrylov),dtype=np.complex_).reshape(dimKrylov,n))
        
        self.krylovr[:,0] = phi0
        self.krylovl[0,:] = phi0.H
        alpha0 =  sp(phi0,H.dot(phi0))
        self.alpha.append(alpha0)
        self.beta.append(0)
        self.gamma.append(0)
        self.dimVal = 1

        # first iteration
        unot = np.matrix(np.zeros(n).reshape(n,1))
        bunot = np.matrix(np.zeros(n).reshape(1,n))
        self.__iterate(self.krylovr[:,0],unot,self.krylovl[0,:],bunot)
        self.dimVal += 1
        
    def __iterate(self,phi_j,phi_jm1,chi_j,chi_jm1):
        phi_jp1,chi_jp1,alpha_jp1,beta_jp1,gamma_jp1=self.recursion(self.H,phi_j,phi_jm1,chi_j,chi_jm1,self.alpha[-1],self.beta[-1],self.gamma[-1])
        self.krylovr[:,self.dimVal] = phi_jp1
        self.krylovl[self.dimVal,:] = chi_jp1
        self.alpha.append(alpha_jp1)
        self.beta.append(beta_jp1)
        self.gamma.append(gamma_jp1)
        
    def iterate(self,niter=None):
        # generic iteration
        nit=niter if niter is not None else self.dimKrylov
        while self.dimVal < nit :
            self.__iterate(self.krylovr[:,self.dimVal-1],self.krylovr[:,self.dimVal-2],self.krylovl[self.dimVal-1,:],self.krylovl[self.dimVal-2,:])
            self.dimVal+=1
            print('Dimension Krylov space = ',self.dimVal,'Orthogonality check = ',ortCheck(self.krylovl,self.krylovr,self.dimVal),'Tridiagonal check = ',tridiagCheck(self.H,self.krylovl,self.krylovr,self.alpha,self.beta,self.gamma,self.dimVal))
