

#This modyle contains the method needed for the analysis of the completeness relation and much more...
import numpy as np
import matplotlib.pyplot as plt

def evalS_nvirt(data):
    """
    The input represents the list of data for a given value of rmult and for a given direction of the field
    """
    coeff_occ = data.log['<psi_i|psi_j>'] 
    coeff_occ=np.double(np.array(coeff_occ))
    
    coeff_vrt = data.log['<psiv_i|D psi_j>']
    coeff_vrt=np.double(np.array(coeff_vrt))
    
    n_occ,n_vrt = coeff_vrt.shape
    print 'no_occ',n_occ, 'n_vrt', n_vrt
    
    sum_occ = 0.0
    for p in range(n_occ):
        for q in range(n_occ):
            sum_occ += coeff_occ[p][q]**2
    
    R = n_occ - sum_occ
    print 'R = ', R
    
    S_nvirt = []
    s = 0.0
    for alpha in range(n_vrt):
        for q in range(n_occ):
            s += coeff_vrt[q][alpha]**2
        S_nvirt.append(R-s)
	
    # rescale by the number of occupied orbitals
    #S_nvirt = [x/n_occ for x in S_nvirt]
    S_nvirt = [x/R for x in S_nvirt]	
    
    return S_nvirt


def validate_eigensystem(H,e,w):
    """
    Inspect the validity of the eigensystem e,w as eigenvalues and 
    eigenvectors of the matrix H    
    """
    for ind,e in enumerate(e):
        isok = np.allclose(H.dot(w[:,ind]),e*w[:,ind],atol=1.e-12)
        if not isok: print('Eigenvalue',ind,'is false')
    print('Validation terminated')

def transition_indexes(np,nalpha,indexes):
    """
    Returns the list of the indices in the bigdft convention that correspond to the 
    couple iorb-ialpha with given spin.
    paramerers: np = tuple of (norbu,norbd) occupied orbitals: when of length 1 assumed spin averaged
                nalpha = tuple of (norbu, norbd)virtual orbitals: when of length 1 assumed spin averaged
                indexes = list of tuples of (iorb,ialpha,ispin) desired indices in python convention (start from 0)
    """
    nspin=len(np)
    inds=[]
    for iorb,ialpha,ispin in indexes:
        jspin=ispin if nspin==2 else 0
        ind=ialpha+iorb*nalpha[jspin] #local index of the spin subspace
        if ispin==1: ind+=np[0]*nalpha[0] #spin 2 comes after spin one
        inds.append(ind)
    return inds

def collection_indexes(np,nalpha,nvirt_small):
    #ugly triple loop
    harvest=[]
    for ispin in [0,1]:
        jspin=ispin if len(np)==2 else 0
        for ip in range(np[jspin]):
            for ialpha in range(nvirt_small[jspin]):
                harvest.append([ip,ialpha,ispin])
    return harvest

def extract_subset(np,nalpha,Cbig,Dbig,nvirt_small):
    """
    Extract from a large Coupling Matrix a submatrix that only consider a subset of the original vectors.
    Use the convention of the transition_indices function for np and nalpha and nvirt_small
    """
    import numpy
    harvest=collection_indexes(np,nalpha,nvirt_small)
    inds=numpy.array(transition_indexes(np,nalpha,harvest))
    return numpy.array([row[inds] for row in Cbig[inds]]),numpy.array(Dbig[inds])

def weight(numOrb,nalpha,exc,eigenproblems, writeRes = True):
    """
    Compute the contribution of the virtual orbitals to the eigenvectors of the coupling matrix
    nalpha = list with numbers of virtual states considered
    exc = list with the index of the eigenvectors considered
    weightP and weightAlpha are structured list. weightP[alpha][i] and weightAlpha[alpha][i] contain 
    the contribution of all the occupied and virtual orbitals (of the system with alpha virtual states) 
    to C_E2[i], respectively
    """
    weightAlpha = []
    for na in nalpha: 
        weight = []
        for excInd in exc:
            alphaProj = np.zeros(na)
            for alpha in range(na):
                # sum over all the occupied orbital and spin 
                indexes = []
                for p in range(numOrb):
                    for spin in [0,1]:
                        indexes.append([p,alpha,spin])
                # extract the value of the index of C_E2
                elements = transition_indexes([numOrb],[na],indexes)
                for el in elements:
                    alphaProj[alpha] += eigenproblems[na][2][excInd][el]**2
            weight.append(alphaProj)
        weightAlpha.append(weight)
        
    weightP = []
    for na in nalpha:
        weight =[]
        for excInd in exc:
            pProj = np.zeros(numOrb)
            for p in range(numOrb):
                # sum over all the virtual orbital and spin 
                indexes = []
                for alpha in range(na):
                    for spin in [0,1]:
                        indexes.append([p,alpha,spin])
                # extract the value of the index of C_E2
                elements = transition_indexes([numOrb],[na],indexes)
                for el in elements:
                    pProj[p] += eigenproblems[na][2][excInd][el]**2
            weight.append(pProj)
        weightP.append(weight)
    
    if writeRes:
        for inda,na in enumerate(nalpha):
            print 'nalpha = ', na
            print ''
            for ind, excInd in enumerate(exc):
                print 'Excitation number :', excInd+1, ' energy = ', 27.211*np.sqrt(eigenproblems[na][1][excInd])
                print '  ******* occupied state contribution ********'
                sumOverThresholdP = 0.0 
                for i,a in enumerate(weightP[inda][ind]):
                    if a > 0.1:
                        sumOverThresholdP+=a
                        print '  occupied state :', i+1, ' weight = ', a
                diffeP = 1.0 - sumOverThresholdP
                print '  1 - sumOverThreshold p = ', '%.3e' % diffeP
            
                print '  ******* virtual state contribution *********'
                sumOverThresholdA = 0.0        
                for i,a in enumerate(weightAlpha[inda][ind]):
                    if a > 0.1:
                        sumOverThresholdA+=a
                        print '  virtual state  :', i+1, ' weight = ', a
                diffeA = 1.0 - sumOverThresholdA
                print '  1 - sumOverThreshold alpha = ', '%.3e' % diffeA
                print ''
    
    return weightP,weightAlpha

def findTransition(wP,wAlpha,threshold = 0.1):
    """
    For each na and excitation index this routine classifies the excitation in term of the 
    states mainly involved (above the threshold)
    """
    pVal = np.where(wP > threshold)[0]
    alphaVal = np.where(wAlpha > threshold)[0]
    tr = ''
    for p in pVal:
        tr+=str(p+1)
    tr+=str('to')
    for p in alphaVal:
        tr+=str(p+1)
    
    return tr

def weightCut(w, threshold = 0.1):
    wCut = np.zeros(len(w))
    for i,ww in enumerate(w):
        if ww < threshold:
            wCut[i] = ww
    return wCut


######################### OLD ROUTINES ###################################


def weightOld(numOrb,nalpha,exc,C_E2,E2, writeRes = True):
    """
    Compute the contribution of the virtual orbitals to the eigenvectors of the coupling matrix
    exc = list with the index of the eigenvectors considered
    weightP and weightAlpha are list. weightP[i] and weightAlpha[i] contain the contribution of all 
    the occupied and virtual orbitals to C_E2[i], respectively
    """
    weightAlpha = []
    for excInd in exc:
        alphaProj = np.zeros(nalpha)
        for alpha in range(nalpha):
            # sum over all the occupied orbital and spin 
            indexes = []
            for p in range(numOrb):
                for spin in [0,1]:
                    indexes.append([p,alpha,spin])
            # extract the value of the index of C_E2
            elements = transition_indexes([numOrb],[nalpha],indexes)
            for el in elements:
                alphaProj[alpha] += C_E2[excInd][el]**2
        weightAlpha.append(alphaProj)
        
    weightP = []
    for excInd in exc:
        pProj = np.zeros(numOrb)
        for p in range(numOrb):
            # sum over all the virtual orbital and spin 
            indexes = []
            for alpha in range(nalpha):
                for spin in [0,1]:
                    indexes.append([p,alpha,spin])
            # extract the value of the index of C_E2
            elements = transition_indexes([numOrb],[nalpha],indexes)
            for el in elements:
                pProj[p] += C_E2[excInd][el]**2
        weightP.append(pProj)
    
    if writeRes:
        for ind, excInd in enumerate(exc):
            print 'Excitation number :', excInd+1, ' energy = ', 27.211*np.sqrt(E2[excInd])
            print '  ******* occupied state contribution ********'
            sumOverThresholdP = 0.0 
            for i,a in enumerate(weightP[ind]):
                if a > 0.1:
                    sumOverThresholdP+=a
                    print '  occupied state :', i+1, ' weight = ', a
            diffeP = 1.0 - sumOverThresholdP
            print '  1 - sumOverThreshold p = ', '%.3e' % diffeP
            
            print '  ******* virtual state contribution *********'
            sumOverThresholdA = 0.0        
            for i,a in enumerate(weightAlpha[ind]):
                if a > 0.1:
                    sumOverThresholdA+=a
                    print '  virtual state  :', i+1, ' weight = ', a
            diffeA = 1.0 - sumOverThresholdA
            print '  1 - sumOverThreshold alpha = ', '%.3e' % diffeA
            print ''
    
    return weightP,weightAlpha


def completeness_relation_new(data):
    """
    Evalute the completenss of the expansion of the pertubed occupied orbitals on the basis of the (occupied and virtual)
    unperturbed ones. The input represents the list of data for a given value of rmult and for a given direction of the 
    field
    """
    coeff_occ = data.log['<psi_i|psi_j>'] 
    coeff_occ=np.double(np.array(coeff_occ))
    
    coeff_vrt = data.log['<psiv_i|D psi_j>']
    coeff_vrt=np.double(np.array(coeff_vrt))
    
    n_occ,n_vrt = coeff_vrt.shape
    print 'no_occ',n_occ, 'n_vrt', n_vrt
    en = data.evals[0][0]
    e_v=[]
    e_o=[]
    for o in range(n_occ):
        e_o.append(en[o])
    for v in range(n_occ,n_occ+n_vrt):
        e_v.append(en[v])

    # compute the norm of the perturbed orbitals projected on the basis of the occupied states
    psiprime=np.array([ 0.0 for i in range(n_occ)])
    for o in range(n_occ):
        psiprime += coeff_occ[o]**2
    # we quantify the magnitude of the missing part
    psiprime = 1.0-psiprime
    print 'sqrt(1-psiprime)',np.sqrt(psiprime) 
    # and we find the maximum value
    reference=np.max(psiprime)
    # we add the contribution of the empty orbitals
    cr=[[] for p in range(n_occ)]
    for p in range(n_occ):
        for v in range(n_vrt):
            psiprime[p] -= coeff_vrt[p][v]**2
            cr[p].append(psiprime[p]/reference)
    return e_v,cr

def crplot_new(e_v,cr,label1,rhoPlot=True, legendPlot=False):
 
    if rhoPlot:
        sm=0.0
        nval=0
        for coeff in cr:
            sm+=np.array(coeff)
            nval+=1
        plt.semilogy(27.211*np.array(e_v),sm/nval,'-',label='Rho')
    else:
        for p,coeff in enumerate(cr):
            plt.semilogy(27.211*np.array(e_v),cr[p],'-',label='Orb_'+str(p))
    if legendPlot:
        plt.legend()
    plt.title('Completeness relation '+label1, fontsize=14)
