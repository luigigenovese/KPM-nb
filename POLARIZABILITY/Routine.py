

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

def solveEigenProblems(numOrb,virtMax,Cmat,dipoles,nalpha):
    """
    Build the dictionary with the solutions of the eigenproblems for each choice of na
    We perform the transpose of the matrix with eigenvectors to have them sorted as row vectors
    """
    eigenproblems = {}
    for na in nalpha:
        C_ext,dipoles_ext=extract_subset([numOrb],[virtMax],Cmat,dipoles,[na])
        print C_ext.shape
        E2,C_E2 = np.linalg.eigh(C_ext)
        C_E2 = C_E2.T
    	eigenproblems[na] = {'Cmat':C_ext,'eigenvalues':E2,'eigenvectors':C_E2,'dipoles':dipoles_ext}
    
    return eigenproblems

def evalOscStrenght(eigenproblems):
    """
    return f[na][eigenvalue index][x_i component] that contains the square
    of the oscillator strenght
    """
    f = {}
    
    for na,e in eigenproblems.iteritems():
        f[na] = []
        numEigenvalues = len(e['eigenvalues'])
        for i in range(numEigenvalues):
            f[na].append(np.dot(e['eigenvectors'][i],e['dipoles'])**2)
    
    return f 

def evalOscStrenghtAvg(eigenproblems):
    """
    return fAvg[na][eigenvalue index] that contains the spatial average
    of the oscillator strenght
    """
    fAvg = {}
    f = evalOscStrenght(eigenproblems)
    
    for na,e in eigenproblems.iteritems():
        fAvg[na] = []
        numEigenvalues = len(e['eigenvalues'])
        for i in range(numEigenvalues):
            s = 0.0
            for x in range(3):
                s+=f[na][i][x]/3.0
            fAvg[na].append(s)
    
    return fAvg       

def evalStatPol(eigenproblems):
    """
    return statPol[na] that contains the vector of the statical polarizability computed
    with na virtual orbitals
    """    
    statPol = {}
    
    f = evalOscStrenght(eigenproblems)
    for na,e in eigenproblems.iteritems():
        alpha = []
        for x in range(3):
            val = 0.0
            E2 = e['eigenvalues']
            for i in range(len(E2)):
                val+= 2.0*f[na][i][x]/E2[i]
            alpha.append(val)
        statPol[na] = alpha
    
    return statPol

def evalSpectrum(eigenproblems,nalphaPlot, domega = 0.005, eta = 1.0e-2):
    """
    return a dictionary with the values of omega (in eV) and the real and
    imaginary part of the spectrum for each value of na
    """
    spectrum = {}
    fAvg = evalOscStrenghtAvg(eigenproblems)
    #fAvg = evalOscStrenght(eigenproblems) # only z component
    
    for na in nalphaPlot:
        if na in eigenproblems.keys():
            spectrum[na] = {}
            omegaMax = np.sqrt(eigenproblems[na]['eigenvalues'][-1])
            npoint = int(omegaMax/domega)
            print 'for na = ', na, ' numpoint = ', npoint, ' omegaMax = ', omegaMax
            omega = np.linspace(0.0,omegaMax,npoint)
            spectrum[na]['omega'] = 27.211*omega
        
            sp = np.zeros(npoint,dtype=np.complex)
            for ind,o in enumerate(omega):
                for i,E in enumerate(eigenproblems[na]['eigenvalues']):
                    sp[ind]+=2.0*fAvg[na][i]/(complex(o,2*eta)**2-E)
                    #sp[ind]+=2.0*fAvg[na][i][2]/(complex(o,2*eta)**2-E) #only z component
            spectrum[na]['realPart'] = -np.real(sp)
            spectrum[na]['imagPart'] = -np.imag(sp)
    
    return spectrum

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
                    alphaProj[alpha] += eigenproblems[na]['eigenvectors'][excInd-1][el]**2
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
                    pProj[p] += eigenproblems[na]['eigenvectors'][excInd-1][el]**2
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
        val = {}
        for i,e in enumerate(exc):
            tr = findTransition(weightP[a][i],weightAlpha[a][i])
            ind=0
            while tr+'-'+str(ind) in val:
                ind+=1
            tr=tr + '-' + str(ind)
            val[tr] = {'weightP' : weightP[a][i], 'weightAlpha' : weightAlpha[a][i], 'level' : [e], 'energy' : 27.211*np.sqrt(eigenproblems[na]['eigenvalues'][e-1]) }
        
        excitations[na] = val
        
    return excitations

def removeDegenarices(excitations,degTol = 1.e-4): 
    # removes the degeneracies (looking for pairs of states with the same energy)
    for na,e in excitations.iteritems():
        engs = []
        for k,v in e.iteritems():
            engs.append([k,v['energy']])
    
        for i in range(len(engs)-1):
            for j in range(i+1,len(engs)):
                if np.allclose(engs[i][1],engs[j][1],degTol) and engs[i][0] in e.keys() and engs[j][0] in e.keys():
                    trnew = engs[i][0]+'+'+engs[j][0]
                    wP = 0.5 * (e[engs[i][0]]['weightP'] + e[engs[j][0]]['weightP'])
                    wA = 0.5 * (e[engs[i][0]]['weightAlpha'] + e[engs[j][0]]['weightAlpha'])
                    eng = 0.5 * (engs[i][1]+engs[j][1])
                    levi = e[engs[i][0]]['level'][0]
                    levj = e[engs[j][0]]['level'][0]
                    excitations[na][trnew] = {'weightP' : wP, 'weightAlpha' : wA, 'level' : [levi,levj], 'energy' : eng }
                    del excitations[na][engs[i][0]]
                    del excitations[na][engs[j][0]]
    
    return excitations

def allTransitions(excitations):
    allTr = []
    nalpha = []
    for na,e in excitations.iteritems():
        nalpha.append(na)
        for ind in e:
            allTr.append(ind)
    allTr=list(set(allTr))
    #print 'Number of distinct transisitons = ', len(allTr)
    # check that nalpha[-1] correspond to the highest value of nalpha
    if nalpha[-1] != max(nalpha): print 'PROBLEM WITH NALPHA LIST SORT'
    
    # remove the transitions that do not appear for all the values of nalpha
    remTrans = 0
    for tr in allTr[::-1]:
        appear = True
        for na in nalpha:
            if tr not in excitations[na]: 
                appear = False
        if appear == False:
            allTr.remove(tr)
            remTrans+=1
    #print 'Number of transisitons removed = ', remTrans   
    
    # sort the transitions according to their energy
    eng = []
    for tr in allTr:
        eng.append(excitations[nalpha[-1]][tr]['energy'])
        
    eng =np.array(eng)
    sortind = np.argsort(eng)
    allTr = [allTr[s] for s in sortind]
    eng = [eng[s] for s in sortind]
    
    return allTr

def pltTrLevel(selTr,excitations,Data,numOrb,plotEng = True):
    for s in selTr:
        for na, e in excitations.iteritems():
            for tr,val in e.iteritems():
                if s in val['level']:
                    if plotEng:
                       plt.scatter(engMax(Data,numOrb,na),val['energy'])
                       plt.annotate(tr,xy=(engMax(Data,numOrb,na),val['energy']))
                    else: 
                       plt.scatter(na,val['energy'])
                       plt.annotate(tr,xy=(na,val['energy']))
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
            for tr,v in e.iteritems():
                if s == tr:
                    val.append(v['energy']) 
        plt.plot(alpha,val)
        plt.scatter(alpha,val,label=s)
        plt.legend(loc=(1.1,0))

def evalSob(numVirtBound,excitations):
    nalpha = excitations.keys()
    nalpha.sort()
    #print nalpha
    
    sob = {}
    allTr = allTransitions(excitations)
    for tr in allTr:
        sobNa = []
        for na in nalpha:
            wA = excitations[na][tr]['weightAlpha']
            sumVal = 0.0
            for ind in range(numVirtBound):
                sumVal+=wA[ind]
            sobNa.append(1.0-sumVal)
        sob[tr] = sobNa
    
    return sob

def evalSobStability(sob,nalpha):
    sobStability = {}
    for tr,s in sob.iteritems():
        val = s[-1]
        deriv = (s[-1]-s[-2])/(nalpha[-1]-nalpha[-2])
        sobStability[tr] = {'value': val, 'derivative':deriv}
    
    return sobStability

def weightCut(w, threshold = 0.1):
    wCut = np.zeros(len(w))
    for i,ww in enumerate(w):
        if ww < threshold:
            wCut[i] = ww
    return wCut

def weightAlphaPlot(selexc,excitations,Data,numOrb,plotEng = True):
    offs = 0.0
    for na, e in excitations.iteritems():
        if selexc in e.keys():
            if plotEng:
                alpha = []
                for a in range(1,na+1):
                    alpha.append(27.211*Data.evals[0][0][numOrb + a -1])
            else:
                alpha = np.linspace(1,na,na)
            wCut = weightCut(e[selexc]['weightAlpha'])
            plt.plot(alpha,offs+wCut, label = 'Nalpha='+str(na))
            offs+=1.2*max(wCut)
    plt.title('Transition '+selexc, fontsize = 14)
    plt.legend(loc=(1.1,0.0))   

######################### OLD ROUTINES ###################################

def energyStableTransitions(excitations, stableTol = 1e-4):
    allTr = allTransitions(excitations)
    
    energyStableTr = {}
    for tr in allTr:
        engTr = []
        for na in excitations:
            engTr.append(excitations[na][tr]['energy'])
        deltaE = max(engTr) - min(engTr)
        if deltaE < stableTol:
            energyStableTr[tr] = {'energy': 0.5*(max(engTr) + min(engTr)),'deltaEnergy' : deltaE}
    
    return energyStableTr

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

