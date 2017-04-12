#example of the module
import numpy as np

def norm(u):
    out = np.sqrt(np.trace(u.T.dot(u)))
    return out

def sp(a,b): #scalar product bra ket
    out = np.trace(np.matmul(a,b))
    return out

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
        u0dotv = np.trace(u0.T.dot(v))
        G+=(u0dotv**2)/(omega-E+1j*eta)
    G = -1/np.pi*G.imag
    return G

def lanc_recursion(H,phis,phism1,alphas,betasp1,betas):
    phisp1 = (H.dot(phis)-alphas*phis-betas*phism1)/betasp1
    Hphisp1 = H.dot(phisp1)
    alphasp1 = np.trace(np.matmul(phisp1.T,Hphisp1))
    betasp2 = norm(Hphisp1-alphasp1*phisp1-betasp1*phis)
    return phisp1,alphasp1,betasp2

def krylovCheck(H,krylov,alpha,beta,dimVal):
    Hkrylov = np.matmul(krylov.T,H.dot(krylov))
    HkrylovReduced = Hkrylov[0:dimVal,0:dimVal]
    Up = np.matrix(np.zeros(dimVal**2).reshape(dimVal,dimVal))
    T = np.matrix(np.diag(alpha))
    for i in range(dimVal-1):
        Up[i,i+1] = beta[i+1]
    T += Up + Up.T
    isok = np.allclose(np.ravel(HkrylovReduced),np.ravel(T),atol=1.e-10)
    return isok

def cont_frac(alpha,beta,omega,eta=1.e-2):
    alphabeta=[(a,b) for a,b in zip(alpha,beta)]
    f=0.0j
    for a,b in reversed(alphabeta):
        f = 1.0/(omega + 1j*eta - a - (b**2)*f)
    return f

def Gfrac(alpha,beta,omega,eta=1.e-2):
    betacut = beta[1:len(beta)]
    frac = cont_frac(alpha,betacut,omega,eta)
    out = -1/np.pi*frac.imag
    return out

def D(i,H,omega,eta=1.e-2):
    DimH = len(H)
    Hsub = H[i:DimH,i:DimH]
    dimHsub = len(Hsub)
    Di = (omega+1j*eta)*np.eye(dimHsub)-Hsub
    Didet = np.linalg.det(Di)
    return Didet

def G(H,omega,eta=1.e-2):
    D0det = D(0,H,omega,eta)
    D1det = D(1,H,omega,eta)
    ratio = D1det/D0det
    out = -1/np.pi*ratio.imag
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
        alpha0 = np.trace(np.matmul(phi0.T,self.H.dot(phi0)))
        self.alpha.append(alpha0)
        self.beta.append(0)
        beta1 = norm_(self.H.dot(phi0)-alpha0*phi0)
        self.beta.append(beta1)
        self.dimVal = 1 #actual dimension of Krylov space

        # first iteration
        unot = np.matrix(np.zeros(n).reshape(n,1)) #needed for the first step
        self.__iterate(self.krylov[:,0],unot)
        self.dimVal += 1       
    def __iterate(self,phi_n,phi_nm1):
        phi1,alpha1,beta2 = self.recursion(self.H,phi_n,phi_nm1,
                                       self.alpha[-1],self.beta[-1],self.beta[-2])
        self.krylov[:,self.dimVal] = phi1
        self.alpha.append(alpha1)
        self.beta.append(beta2)
    def iterate(self,niter=None):
        # generic iteration
        nit=niter if niter is not None else self.dimKrylov
        while self.dimVal < nit :
            self.__iterate(self.krylov[:,self.dimVal-1],self.krylov[:,self.dimVal-2])
            self.dimVal+=1

