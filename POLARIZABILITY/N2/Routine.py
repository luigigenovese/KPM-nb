#This modyle contains the method needed for the analysis of the completeness relation
import numpy as np
import matplotlib.pyplot as plt

def completeness_relation(data):
    """
    Evalute the completenss of the expansion of the pertubed occupied orbitals on the basis of the (occupied and virtual)
    unperturbed ones. 
    """
    coeff_occ = [n.log['<psi_i|psi_j>'] for n in data]
    coeff_occ=np.double(np.array(coeff_occ))
    
    coeff_vrt = [n.log['<psiv_i|D psi_j>'] for n in data]
    coeff_vrt=np.double(np.array(coeff_vrt))
    
    n_occ,n_vrt = coeff_vrt[0].shape
    print 'no_occ',n_occ, 'n_vrt', n_vrt
    en = data[0].evals[0][0]
    e_v=[]
    e_o=[]
    for o in range(n_occ):
        e_o.append(en[o])
    for v in range(n_occ,n_occ+n_vrt):
        e_v.append(en[v])

    # compute the norm of the perturbed orbitals projected on the basis of the occupied states
    psiprimeEx=np.array([ 0.0 for i in range(n_occ)])
    psiprimeEz=np.array([ 0.0 for i in range(n_occ)])
    for o in range(n_occ):
        psiprimeEx += coeff_occ[0][o]**2
        psiprimeEz += coeff_occ[1][o]**2
    # we quantify the magnitude of the missing part
    psiprimeEx = 1.0-psiprimeEx
    psiprimeEz = 1.0-psiprimeEz
    print 'psiprimeEx Norm ',np.sqrt(psiprimeEx)
    print 'psiprimeEz Norm ',np.sqrt(psiprimeEz)
    # and we find the maximum value
    referenceEx=np.max(psiprimeEx)
    referenceEz=np.max(psiprimeEz)
    # we add the contribution of the empty orbitals
    crEx=[[] for p in range(n_occ)]
    crEz=[[] for p in range(n_occ)]
    for p in range(n_occ):
        for v in range(n_vrt):
          psiprimeEx[p] -= coeff_vrt[0][p][v]**2
          psiprimeEz[p] -= coeff_vrt[1][p][v]**2    
          crEx[p].append(psiprimeEx[p]/referenceEx)
          crEz[p].append(psiprimeEz[p]/referenceEz)
    return e_v,crEx,crEz

def crplot(e_v,cr,label1,label2,rhoPlot=True):
 
    if rhoPlot:
        sm=0.0
        nval=0
        for coeff in cr:
            sm+=np.array(coeff)**2
            nval+=1
        plt.semilogy(27.211*np.array(e_v),sm/nval,'-',label='Rho_'+label2)
    else:
        for p,coeff in enumerate(cr):
            plt.semilogy(27.211*np.array(e_v),cr[p],'-',label='Orb_'+str(p)+'_'+label2)
    plt.legend()
    plt.title('Completeness relation '+label1, fontsize=14)
