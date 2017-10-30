

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
    exc = list with the excitations considered (1 for the first excitation and so on...) 
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
                    alphaProj[alpha] += eigenproblems[na][2][excInd-1][el]**2
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
                    pProj[p] += eigenproblems[na][2][excInd-1][el]**2
            weight.append(pProj)
        weightP.append(weight)
    
    if writeRes:
        for inda,na in enumerate(nalpha):
            print 'nalpha = ', na
            print ''
            for ind, excInd in enumerate(exc):
                print 'Excitation level :', excInd, ' energy = ', 27.211*np.sqrt(eigenproblems[na][1][excInd-1])
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

def engMax(Data,numOrb,na):
    em = 27.211*Data.evals[0][0][numOrb + na -1]
    return em

def findTransition(wP,wAlpha,threshold = 0.1):
    """
    For each na and excitation index this routine classifies the excitation in term of the 
    states mainly involved (above the threshold)
    """
    pVal = np.where(wP > threshold)[0]
    alphaVal = np.where(wAlpha > threshold)[0]
    tr = ''
    for p in pVal:
        tr+=str(p+1)+str(',')
    tr=tr[:-1]
    tr+=str('to')
    for p in alphaVal:
        tr+=str(p+1)+str(',')
    tr=tr[:-1]

    return tr

def buildExcitations(numOrb,nalpha,exc,eigenproblems): 
    excitations = {}
    weightP,weightAlpha = weight(numOrb,nalpha,exc,eigenproblems, False)
 
    for a,na in enumerate(nalpha):
        transitions = {}
        for i,e in enumerate(exc):
            tr = findTransition(weightP[a][i],weightAlpha[a][i])
            ind=0
            while tr+'-'+str(ind) in transitions:
                ind+=1
            tr=tr + '-' + str(ind)
            transitions[tr] = {'weightP' : weightP[a][i], 'weightAlpha' : weightAlpha[a][i], 'level' : [e], 'eng' : 27.211*np.sqrt(eigenproblems[na][1][e-1]) }
        excitations[na] = {'Cmat' : eigenproblems[na][0], 'E2': eigenproblems[na][1], 'C_E2' : eigenproblems[na][2], 'transitions' : transitions}
    
    return excitations

def removeDegenarices(excitations,degTol = 1.e-4): 
    # removes the degeneracies (looking for pairs of states with the same energy)
    for na,e in excitations.iteritems():
        engs = []
        for k,v in e['transitions'].iteritems():
            engs.append([k,v['eng']])
    
        for i in range(len(engs)-1):
            for j in range(i+1,len(engs)):
                if np.allclose(engs[i][1],engs[j][1],degTol) and engs[i][0] in e['transitions'].keys() and engs[j][0] in e['transitions'].keys():
                    trnew = engs[i][0]+'+'+engs[j][0]
                    wP = 0.5 * (e['transitions'][engs[i][0]]['weightP'] + e['transitions'][engs[j][0]]['weightP'])
                    wA = 0.5 * (e['transitions'][engs[i][0]]['weightAlpha'] + e['transitions'][engs[j][0]]['weightAlpha'])
                    eng = 0.5 * (engs[i][1]+engs[j][1])
                    levi = e['transitions'][engs[i][0]]['level'][0]
                    levj = e['transitions'][engs[j][0]]['level'][0]
                    excitations[na]['transitions'][trnew] = {'weightP' : wP, 'weightAlpha' : wA, 'level' : [levi,levj], 'eng' : eng }
                    del excitations[na]['transitions'][engs[i][0]]
                    del excitations[na]['transitions'][engs[j][0]]
    
    return excitations

def allTransitions(excitations):
    allTr = []
    for e in excitations.values():
        for ind in e['transitions']:
            allTr.append(ind)
    allTr=list(set(allTr))
    
    eng = []
    for a in allTr:
        notCounted = True
        for na,e in excitations.iteritems():
            if notCounted:
                for ind,v in e['transitions'].iteritems():
                    if a == ind:
                        eng.append(v['eng'])
                        notCounted = False
                        break
    eng =np.array(eng)
    sortind = np.argsort(eng)
    allTr = [allTr[s] for s in sortind]
    return allTr

def stableTransitions(excitations, stableTol = 1e-4):
    allTr = allTransitions(excitations)
    stableTr = []
    for tr in allTr:
        engTr = []
        for e in excitations.values():
            for ind,v in e['transitions'].iteritems():
                if tr == ind:
                    engTr.append(v['eng'])
        deltaE = max(engTr) - min(engTr)
        if len(engTr) > 1 and deltaE < stableTol:
            stableTr.append([tr,0.5*(max(engTr) + min(engTr)),deltaE])
    return stableTr

def pltTrLevel(selTr,excitations,Data,numOrb,plotEng = True):
    for s in selTr:
        for na, e in excitations.iteritems():
            for tr,val in e['transitions'].iteritems():
                if s in val['level']:
                    if plotEng:
                       plt.scatter(engMax(Data,numOrb,na),val['eng'])
                       plt.annotate(tr,xy=(engMax(Data,numOrb,na),val['eng']))
                    else: 
                       plt.scatter(na,val['eng'])
                       plt.annotate(tr,xy=(na,val['eng']))
    plt.show()

def pltTrLabel(selLab,excitations,Data,numOrb,plotEng = True):
    for s in selLab:
        alpha = []
        val = []
        for na, e in excitations.iteritems():
            if plotEng:
               alpha.append(engMax(Data,numOrb,na))
            else:
               alpha.append(na)
            for tr,v in e['transitions'].iteritems():
                if s == tr:
                    val.append(v['eng'])
        if len(alpha) == len(val):        
           #print s, alpha, val
           plt.plot(alpha,val)
           plt.scatter(alpha,val,label=s)
           plt.legend(loc=(1.1,0))
        else:
           print 'energy of the transition '+str(s)+' not found for all the na'
           print s, alpha, val

def weightCut(w, threshold = 0.1):
    wCut = np.zeros(len(w))
    for i,ww in enumerate(w):
        if ww < threshold:
            wCut[i] = ww
    return wCut

def sotPlot(selLab,excitations):
    sot = {}
    for s in selLab:
        alpha = []
        out = []
        for na,e in excitations.iteritems():
            for ind,v in e['transitions'].iteritems():
                if ind == s:
                    sW = 0.0
                    for w in weightCut(v['weightAlpha']):
                        sW+=w
                    alpha.append(na)
                    out.append(sW)
                    sot[s] = [alpha,out]
                    
    for tr,val in sot.iteritems():
        plt.semilogy(val[0],val[1],label=tr)
    plt.legend(loc=(1.1,0.0))    
    return sot

def sotPlotNorm(sot):
    for tr,val in sot.iteritems():
        plt.plot(val[0],val[1]/val[1][0],label=tr)
    plt.legend(loc=(1.1,0.0))   

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
