

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

def alphaWeight(numOrb,nalpha,numExc,C_E2,E2, writeRes = True):
    """
    Compute the contribution of the virtual orbitals to the eigenvectors of the coupling matrix
    numExc = number of eigenvectors considered: from the lowest one up to numExc-1
    weight is a list. weight[i] contains the contribution of all the virtual orbitals to C_E2[i]		 
    """
    weight = []
    for excInd in range(numExc):
        alphaProj = [0.0 for i in range(nalpha)]
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
        weight.append(alphaProj)
    
    if writeRes:
        for excInd in range(numExc):
            print 'Exctation number :', excInd+1, ' energy = ', np.sqrt(E2[excInd])
	    sumOverThreshold = 0.0        
	    for i,a in enumerate(weight[excInd]):
                if a > 0.1:
                    sumOverThreshold+=a
                    print '  virtual state :', i+1, ' weight = ', a
            diffe = 1.0 - sumOverThreshold
            print '1 - sumOverThreshold = ', '%.3e' % diffe
            print ''
    return weight


######################### OLD ROUTINES ###################################

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
