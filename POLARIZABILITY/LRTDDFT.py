
import numpy as np
from futile.Utils import write
HaeV=27.21138386

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

def solveEigenProblems(numOrb,box,nalpha):
    """
    Build the dictionary with the solutions of the eigenproblems for each choice of na
    We perform the transpose of the matrix with eigenvectors to have them sorted as row vectors
    """
    eigenproblems = {}
    for na in nalpha:
        if na > box['nvirt']:
            print 'There are not enough virtual states for', na
            continue
        C_ext,dipoles_ext=extract_subset([numOrb],[box['nvirt']],box['C'],box['T'],[na])
        newdipole=[]
        for line in dipoles_ext:
            newdipole.append(line[0]*np.array(line[1:]))
        newdipole=np.array(newdipole)
        E2,C_E2 = np.linalg.eigh(C_ext)
        C_E2 = C_E2.T
        write('Eigensystem solved for',na)
    	eigenproblems[na] = {'Cmat':C_ext,'eigenvalues':E2,'eigenvectors':C_E2,'transitions':newdipole}
    return eigenproblems

def diagonalize_CM(norb,syst,nalpha,addBoxNivrt = True):
    for rVal,box in syst.iteritems():
        write('Solve for rVal = ', rVal)   
	if addBoxNivrt : 
           ep = solveEigenProblems(norb,box,nalpha+[box['nvirt']])
        else :
           ep = solveEigenProblems(norb,box,nalpha)
        box['eigenproblems'] = ep

def get_alpha_energy(log,norb,nalpha):
    return log.evals[0][0][norb+nalpha-1]

def get_p_energy(log,norb):
    return log.evals[0][0][0:norb]

def get_oscillator_strengths(evect,trans):
    scpr=np.dot(evect,trans)
    os=[np.array(t[0:3])**2 for t in scpr]
    return np.array(os)

def get_oscillator_strenght_avg(os):
    os_avg = []
    for vet in os:
        avg = 0.0
        for comp in vet:
            avg+=comp
        os_avg.append(avg/3.0)
    return np.array(os_avg)

def static_polarizabilities(e2,os):
    val=0.0
    for e,f in zip(e2,os):
        val+= 2.0*f/e
    return val

def gather_excitation_informations(dict_casida):
    """
    Gived a Casida's eigeproblem (a diagonalized set of eigenvalues of the casida matrix)
    It provides the information needed to extract absorption spectra and susceptivity-related quantities
    """
    os=get_oscillator_strengths(dict_casida['eigenvectors'],dict_casida['transitions'])
    dict_casida['oscillator_strengths']=os
    dict_casida['oscillator_strength_avg']=get_oscillator_strenght_avg(os)
    dict_casida['alpha_xyz']=static_polarizabilities(dict_casida['eigenvalues'],os)

def collect_LR(syst):
    for rVal in syst:
        dict_box=syst[rVal]
        for nalpha in dict_box['eigenproblems']:
            dict_casida=dict_box['eigenproblems'][nalpha]
            gather_excitation_informations(dict_casida)

def collect_Alpha(syst,norb):
    """
    Build a dictionary with the statical polarizabilities written in function on nalpha, for
    each choice of rmult
    """
    alpha={}
    for rVal in syst:
        alpha[rVal]={}
        nalpha = syst[rVal]['eigenproblems'].keys()
        nalpha.sort()
        alpha[rVal]['nalpha'] = nalpha
        eng = []
        val=[[],[],[]]
        for na in nalpha:
            eng.append(get_alpha_energy(syst[rVal]['logfile'],norb,na))
            for ind in range(3):
                val[ind].append(syst[rVal]['eigenproblems'][na]['alpha_xyz'][ind])
        alpha[rVal]['naEnergy'] = HaeV*np.array(eng)
        alpha[rVal]['alphaX'] = val[0]
        alpha[rVal]['alphaY'] = val[1]
        alpha[rVal]['alphaZ'] = val[2]
    return alpha

def get_spectrum(e2,f,omegaMax,domega = 0.005,eta = 1.0e-2):
    """
    Given a single Casida solution, return a dictionary with the values of omega (in eV) and the real and
    imaginary part of the spectrum
    """
    spectrum = {}
    npoint = int(omegaMax/domega)
    print 'numpoint = ', npoint, ' omegaMax (eV) = ', HaeV*omegaMax
    omega = np.linspace(0.0,omegaMax,npoint)
    spectrum['omega'] = HaeV*omega
    
    sp = np.zeros(npoint,dtype=np.complex)
    for ind,o in enumerate(omega):
        for ff,e in zip(f,e2):
            sp[ind]+=2.0*ff/((o+1j*2.0*eta)**2-e)
    spectrum['realPart'] = -np.real(sp)
    spectrum['imagPart'] = -np.imag(sp)
    return spectrum

def collect_spectrum(syst,domega = 0.005,eta = 1.0e-2):
    sp = {}
    for rVal in syst:
        nvirt = syst[rVal]['nvirt']
        f = syst[rVal]['eigenproblems'][nvirt]['oscillator_strength_avg']
        e2 = syst[rVal]['eigenproblems'][nvirt]['eigenvalues']
        omegaMax = np.sqrt(e2[-1])
        sp[rVal] = get_spectrum(e2,f,omegaMax,domega,eta)
    return sp

def identify_contributions(numOrb,na,exc,C_E2):
    pProj = np.zeros(numOrb*2)
    for p in range(numOrb):
        for spin in [0,1]:
                # sum over all the virtual orbital and spin 
            for alpha in range(na):                 
                # extract the value of the index of C_E2
                elements = transition_indexes([numOrb],[na],[[p,alpha,spin]])
                for el in elements:
                    pProj[p+numOrb*spin] += C_E2[exc][el]**2
    return pProj

def get_p_energy(log,norb):
    return log.evals[0][0][0:norb]

def get_threshold(pProj,evals,tol):
    norb=len(evals)
    spinup=pProj[0:norb].tolist()
    spindw=pProj[norb:2*norb].tolist()
    spinup.reverse()
    spindw.reverse()
    imax=norb-1
    for valu,vald in zip(spinup,spindw):
        if max(valu,vald) > tol: break
        imax-=1
    return [imax+1,-evals[imax]]

def find_excitation_thr(dict_casida,na,nexc,evals,tol):
    norb=len(evals)
    thrs=[]
    for a in range(nexc):
        proj=identify_contributions(norb,na,a,dict_casida['eigenvectors'])
        th=get_threshold(proj,evals,tol)
        thrs.append(th)
    dict_casida['thresholds']=np.array(thrs)

def collect_excitation_thr(syst,numOrb,numExc,tol=5e-2):
    for rVal in syst:
        nvirt=syst[rVal]['nvirt']
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        #numExc = len(dict_casida['eigenvalues'])
        pEng = get_p_energy(syst[rVal]['logfile'],numOrb)
        find_excitation_thr(dict_casida,nvirt,numExc,pEng,tol)  

def split_channels(dict_casida,numOrb,numExc):
    channels = [[] for i in range(numOrb)]
    for exc in range(numExc):
        ind = np.rint(dict_casida['thresholds'][exc][0]-1)
        channels[ind.astype(int)].append(HaeV*np.sqrt(dict_casida['eigenvalues'][exc]))
    return channels  

def collect_channels(syst,numOrb,numExc):
    channels = {}
    for rVal in syst:
        nvirt = syst[rVal]['nvirt']
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        channels[rVal] = split_channels(dict_casida,numOrb,numExc)
    return channels

def split_excitations_index(dict_casida):
    ind_bt = []
    ind_at = []
    numExc = len(dict_casida['thresholds'])
    for ind in range(numExc):
        if np.sqrt(dict_casida['eigenvalues'][ind]) < dict_casida['thresholds'][ind][1]:
            ind_bt.append(ind)
        else : ind_at.append(ind)
    return ind_bt,ind_at

def collect_spectrum_bt_at_old(syst,domega = 0.005,eta = 1.0e-2):
    sp_bt = {}
    sp_at = {}
    for rVal in syst:
        nvirt = syst[rVal]['nvirt']
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        exc_bt,exc_at,f_bt,f_at = split_excitations_osStrenght(dict_casida)
        omegaMax = np.sqrt(dict_casida['eigenvalues'][-1])
        sp_bt[rVal] = get_spectrum(exc_bt,f_bt,omegaMax,domega,eta)
        sp_at[rVal] = get_spectrum(exc_at,f_at,omegaMax,domega,eta)
    return sp_bt,sp_at

def collect_spectrum_bt_at(syst,domega = 0.005,eta = 1.0e-2):
    sp_bt = {}
    sp_at = {}
    for rVal in syst:
        nvirt = syst[rVal]['nvirt']
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        numExc = len(dict_casida['thresholds'])
        ind_bt,ind_at = split_excitations_index(dict_casida)
        omegaMax = np.sqrt(dict_casida['eigenvalues'][numExc-1])
        sp_bt[rVal] = get_spectrum(dict_casida['eigenvalues'][ind_bt],dict_casida['oscillator_strength_avg'][ind_bt],omegaMax,domega,eta)
        sp_at[rVal] = get_spectrum(dict_casida['eigenvalues'][ind_at],dict_casida['oscillator_strength_avg'][ind_at],omegaMax,domega,eta)
    return sp_bt,sp_at





