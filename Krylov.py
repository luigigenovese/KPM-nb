#This modyle contains the method needed for the Lanczos procedure 
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

def lanc_recursion(H,phij,phijm1,alphaj,betaj):
    phihatjp1 = H.dot(phij)-alphaj*phij-betaj*phijm1
    betajp1 = norm(phihatjp1)
    phijp1 = phihatjp1/betajp1
    alphajp1 = np.trace(np.matmul(phijp1.T,H.dot(phijp1)))
    return phijp1,alphajp1,betajp1

def ortCheck(krylov,dimVal):
    """
    Check the orthonormality of the Krylov vectors
    """
    klkr = np.matmul(krylov.T,krylov)
    klkrReduced = klkr[0:dimVal,0:dimVal]
    isok = np.allclose(klkrReduced,np.matrix(np.eye(dimVal)),atol=1.e-10)
    return isok

def Tmatrix(alpha,beta,dimVal):
    """
    Compute a tridiagonal matrix of dimension (dimVal,dimVal) with elements given by the alpha and
    beta parameters 
    """
    T = np.matrix(np.diag(alpha))
    Treduced = T[0:dimVal,0:dimVal]
    Up = np.matrix(np.zeros(dimVal**2).reshape(dimVal,dimVal))
    for i in range(dimVal-1):
        Up[i,i+1] = beta[i+1]
    Treduced += Up + Up.T
    return Treduced
    
def tridiagCheck(H,krylov,alpha,beta,dimVal):
    """
    Check if the matrix H is tridiagonal, with elements given by the alpha and
    beta parameters, when expressed in the basis of the Krylov vectors
    """
    Hkrylov = np.matmul(krylov.T,H.dot(krylov))
    HkrylovReduced = Hkrylov[0:dimVal,0:dimVal]
    isok = np.allclose(HkrylovReduced,Tmatrix(alpha,beta,dimVal),atol=1.e-10)
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
        alpha0 = np.trace(np.matmul(phi0.T,H.dot(phi0)))
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
            print('Dimension Krylov space = ',self.dimVal,'Orthogonality check = ',ortCheck(self.krylov,self.dimVal),'Tridiagonal check = ',tridiagCheck(self.H,self.krylov,self.alpha,self.beta,self.dimVal))

