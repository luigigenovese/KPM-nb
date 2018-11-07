 
import numpy as np
from futile.Utils import write
HaeV=27.21138386


def _occ_and_virt(log):
    """
    Extract the number of occupied and empty orbitals from a logfile
    """
    if log.log['Spin treatment'] == 'Averaged':
        norb=log.log['Total Number of Orbitals']
        norbv=log.evals[0].info[0]-norb
    else:
        raise ValueError('Coupling matrix reading for spin-polarized system should be implemented')
    return (norb,),(norbv,)

def transition_indexes(np,nalpha,indexes):
    """
    Returns the list of the indices in the bigdft convention that correspond to the 
    couple iorb-ialpha with given spin.
    
    Args:
        np (tuple): (norbu,norbd) occupied orbitals: when of length 1 assumed spin averaged
        nalpha (tuple): (norbu, norbd)virtual orbitals: when of length 1 assumed spin averaged
        indexes (list): list of tuples of (iorb,ialpha,ispin) desired indices in python convention (start from 0)
    """
    nspin=len(np)
    inds=[]
    for iorb,ialpha,ispin in indexes:
        jspin=ispin if nspin==2 else 0
        ind=ialpha+iorb*nalpha[jspin] #local index of the spin subspace
        if ispin==1: ind+=np[0]*nalpha[0] #spin 2 comes after spin one
        inds.append(ind)
    return inds

def _collection_indexes(np,nvirt_small):
    harvest=[]
    for ispin in [0,1]:
        jspin=ispin if len(np)==2 else 0
        for ip in range(np[jspin]):
            for ialpha in range(nvirt_small[jspin]):
                harvest.append([ip,ialpha,ispin])
    return harvest

def _collection_indexes_iocc(iocc,nvirt,spin=None):
    harvest=[]
    for ispin in [0,1]:
        jspin=ispin if len(nvirt)==2 else 0
        if spin is not None and ispin != spin: continue
        for ialpha in range(nvirt[jspin]):
            harvest.append([iocc,ialpha,ispin])
    return harvest


def extract_subset(npts,nalpha,Cbig,Dbig,nvirt_small):
    """
    Extract from a large Coupling Matrix a submatrix that only consider a subset of the original vectors.
    Use the convention of the transition_indices function for np and nalpha and nvirt_small
    """
    harvest=collection_indexes(npts,nvirt_small)
    inds=np.array(transition_indexes(npts,nalpha,harvest))
    return np.array([row[inds] for row in Cbig[inds]]),np.array(Dbig[inds])


class TransitionMatrix(np.matrix):
    """
    Matrix of Transition Quantities, might be either Coupling Matrix or transition dipoles
    
    Args:
        matrix (matrix-like): data of the coupling matrix. If present also the number of orbitals should be provided.
        norb_occ (tuple): number of occupied orbitals per spin channnel. Compulsory if ``matrix`` is specified. 
        norb_virt (tuple): number of empty orbitals per spin channnel. Compulsory if ``matrix`` is specified.
        log (Logfile): Instance of the logfile from which the coupling matrix calculation is performed. 
            Automatically retrieves the ``norb_occ`` and `norb_virt`` parameters.
    """
    def __new__(cls,matrix=None,norb_occ=None,norb_virt=None,log=None):
        """
        Create the object from the arguments and return the ``self`` instance
        """
        import os
        if log is not None:
            datadir=log.log.get('radical','')
            datadir = 'data-'+datadir if len(datadir)>0 else 'data'
            cmfile=os.path.join(log.srcdir,datadir,cls._filename)
            if not os.path.isfile(cmfile):
                raise ValueError('The file "'+cmfile+'" does not exist')
            norb,norbv=_occ_and_virt(log)
            write('Loading data with ',norb,' occupied and ',norbv,' empty states, from file "',cmfile,'"')
            try:
                import pandas as pd
                write('Using pandas:')
                mat=pd.read_csv(cmfile,delim_whitespace=True,header=None)
            except:
                write('Using numpy:')
                mat=np.loadtxt(cmfile)
            write('done')
        else:
            mat=matrix
        return super(TransitionMatrix,cls).__new__(cls,mat)
    def __init__(self,*args,**kwargs):
        """
        Perform sanity checks on the loaded matrix
        """
        log=kwargs.get('log')
        if log is not None:
            self.norb_occ,self.norb_virt=_occ_and_virt(log)
        else:
            self.norb_occ=kwargs.get('norb_occ')
            self.norb_virt=kwargs.get('norb_virt')
        assert(self.shape[0] == self._total_transitions())
        write("Shape is conformal with the number of orbitals")
        self._sanity_check()
    def _total_transitions(self):
        ntot=0
        for no,nv in zip(self.norb_occ,self.norb_virt):
            ntot+=no*nv
        if len(self.norb_occ) == 1: ntot *=2
        return ntot
    def _subindices(self,norb_occ,norb_virt):
        for i,(no,nv) in enumerate(zip(norb_occ,norb_virt)):
            assert(no <= self.norb_occ[i] and nv <= self.norb_virt[i])
        harvest=_collection_indexes(norb_occ,norb_virt)
        return np.array(transition_indexes(norb_occ,self.norb_virt,harvest))   
    def _sanity_check(self):
        pass

class CouplingMatrix(TransitionMatrix):
    """
    Casida Coupling Matrix, extracted from the calculation performed by BigDFT
    """
    _filename='coupling_matrix.txt'
    def _sanity_check(self):
        write('Casida Matrix is symmetric',np.allclose(self,self.T,atol=1.e-12))
    def subportion(self,norb_occ,norb_virt):
        """Extract a subportion of the coupling matrix.
        
        Returns a Coupling Matrix which is made by only considering the first ``norb_occ`` and ``norb_virt`` orbitals
        
        Args:
           norb_occ (tuple): new number of occupied orbitals. Must be lower that the instance value
           norb_virt (tuple): new number of virtual orbitals. Must be lower that the instance value
        """
        inds=self._subindices(norb_occ,norb_virt)
        mat=np.array([row[0,inds] for row in self[inds]])
        return CouplingMatrix(matrix=mat,norb_occ=norb_occ,norb_virt=norb_virt)
    def diagonalize(self):
        """
        Diagonalize the Coupling Matrix
        
        Returns:
            E2, CE2: tuple of the Eigenvvalues and Eigenvectors of the coupling matrix.
               We perform the transpose of the matrix with eigenvectors to have them sorted as row vectors 
        """
        write('Diagonalizing Coupling matrix of shape',self.shape)
        E2,C_E2 = np.linalg.eigh(self)
        write('Eigensystem solved')
        C_E2 = C_E2.T
        return E2,C_E2

class TransitionMultipoles(TransitionMatrix):
    """
    Transition dipoles, extracted from the calculation performed by BigDFT
    """
    _filename='transition_quantities.txt'
    def subportion(self,norb_occ,norb_virt):
        """Extract a subportion of the coupling matrix.
        
        Returns a Coupling Matrix which is made by only considering the first ``norb_occ`` and ``norb_virt`` orbitals
        
        Args:
           norb_occ (tuple): new number of occupied orbitals. Must be lower that the instance value
           norb_virt (tuple): new number of virtual orbitals. Must be lower that the instance value
        """
        inds=self._subindices(norb_occ,norb_virt)
        mat=np.array(self[inds])
        return TransitionMultipoles(matrix=mat,norb_occ=norb_occ,norb_virt=norb_virt)
    def get_transitions(self):
        """
        Get the transition quantities as the dimensional objects which should contribute to the oscillator strengths.
        
        Returns:
            newtransitions (array): Transition quantities multiplied by the square root of the unperturbed transition energy
        """
        newdipole=[]
        for line in self:
            newdipole.append(np.ravel(line[0,0]*line[0,1:]))
        return np.array(newdipole)
    
class TransitionDipoles(TransitionMultipoles):
    """
    Transition dipoles as provided in the version of the code < 1.8.0. Deprecated, to be used in some particular cases
    """
    _filename='transition_dipoles.txt'
    def get_transitions(self):
        return self
    
    
class Excitations():
    """LR Excited states of a system
    
    Definition of the excited states in the Casida Formalism
    
    Args:
       cm (CouplingMatrix): the matrix of coupling among transitions
       tm (TransitionMultipoles): scalar product of multipoles among transitions
       
    Attributes:
        to be described with google docstring syntax
    
    """
    def __init__(self,cm,tm):
        self.cm=cm
        self.tm=tm
        self.eigenvalues,self.eigenvectors=cm.diagonalize()
        self.transitions=tm.get_transitions()
        self._oscillator_strengths()
        
    def _oscillator_strengths(self):
        scpr=np.array(np.dot(self.eigenvectors,self.transitions))
        self.oscillator_strenghts= np.array([t**2 for t in scpr[:,0:3]]) 
        self.avg_os=np.average(self.oscillator_strenghts,axis=1)
        shp=self.eigenvalues.shape[0]
        self.alpha_prime = 2.0*self.oscillator_strenghts/self.eigenvalues[:,np.newaxis]
               
        
    def spectrum_curves(self,omega,slice=None,weights=None):
        """Calculate spectrum curves.
        
        Provide the set of the curves associated to the weights. The resulting curves might then be used to draw the excitation spectra.
        
        Args:
            omega (array): the linspace used for the plotting, of shape ``(n,)``. Must be provided in Atomic Units
            slice (array): the llokup array that has to be considered. if Not provided the entire range is assumed
            weights (array): the set of arrays used to weight the spectra. Must have shape ``(rank,m)``, where ``rank`` is equal to the number of eigenvalues.
                If m is absent it is assumed to be 1. When not specified, it defaults to the average oscillator strenghts.
            
        Returns:
            array: a set of spectrum curves, of shape equal to ``(n,m)``, where ``n`` is the shape of ``omega`` and ``m`` the size of the second dimension of ``weights``.
        """
        if slice is None:
            oo=self.eigenvalues[:,np.newaxis] - omega**2
            wgts=weights if weights is not None else self.avg_os
        else:
            oo=self.eigenvalues[slice,np.newaxis] - omega**2
            oo=oo[0]
            wgts=weights if weights is not None else self.avg_os[slice]
        return np.dot(2.0/oo.T,wgts)
    
    def _project_on_occ(self,exc):
        """
        Project a given eigenvalue on the occupied orbitals.
        In the spin averaged case consider only half of the orbitals
        """
        norb_occ=self.cm.norb_occ
        norb_virt=self.cm.norb_virt
        pProj_spin=[]
        for ispin,norb in enumerate(norb_occ):
            pProj=np.zeros(norb)
            for iorb in range(norb):
                harvest=_collection_indexes_iocc(iorb,self.cm.norb_virt,spin=None if len(norb_occ) == 1 else ispin)
                inds=np.array(transition_indexes(norb_occ,norb_virt,harvest))
                pProj[iorb]=np.sum(np.ravel(self.eigenvectors[exc,inds])**2)
            pProj_spin.append(pProj)
        return pProj_spin
    
    def _get_threshold(self,pProj_spin,th_energies,tol):
        """
        Identify the energy which is associated to the threshold of a given excitation.
        The tolerance is used to discriminate the component
        """
        ths=-1.e100
        for proj,en in zip(pProj_spin,th_energies):
            norb=len(en)
            pProj = proj.tolist()
            pProj.reverse()
            imax=norb-1
            for val in pProj:
                if val > tol: break
                imax-=1
            ths=max(ths,en[imax])
        return ths
    
    def split_excitations(self,evals,tol,nexc='all'):
        """Separate the excitations in channels.
        
        This methods classify the excitations according to the channel they belong, and determine if a 
        given excitation might be considered as a belonging to a discrete part of the spectrum or not.
        
        Args:
            evals (BandArray): the eigenvalues as they are provided (for instance) from a `Logfile` class instance.
            tol (float): tolerance for determining the threshold
            nexc (int,str): number of excitations to be analyzed. If ``'all'`` then the entire set of excitations are analyzed.

        """
        self.determine_occ_energies(evals)
        self.identify_thresholds(self.occ_energies,tol,len(self.eigenvalues) if nexc == 'all' else nexc)
        
    
    def identify_thresholds(self,occ_energies,tol,nexc):
        """Identify the thresholds per excitation.
        
        For each of the first ``nexc`` excitations, identify the energy value of its corresponding threshold.
        This value is determined by projecting the excitation components on the occupied states and verifying that
        their norm for the highest energy level is below a given tolerance.
        
        Args:
           occ_energies (tuple of array-like): contains the list of the energies of the occupied states per spin channel
           tol (float): tolerance for determining the threshold
           nexc (int): number of excitations to be analyzed
        """
        self.wp_norms=[] #: Norm of the $w_p^a$ states associated to each excitation
        threshold_energies=[]
        for exc in range(nexc):
            proj=self._project_on_occ(exc)
            self.wp_norms.append(proj)
            threshold_energies.append(self._get_threshold(proj,occ_energies,tol))
        self.threshold_energies=np.array(threshold_energies) #: list: identified threshold for inspected excitations
        
        self.excitations_below_threshold=np.where(np.abs(self.threshold_energies) >= np.sqrt(self.eigenvalues[0:len(self.threshold_energies)]))
        """ array: Indices of the excitations which lie below their corresponding threshold """
        
        self.excitations_above_threshold=np.where(np.abs(self.threshold_energies) < np.sqrt(self.eigenvalues[0:len(self.threshold_energies)]))
        """ array: Indices of the excitations which lie above their corresponding threshold """
        
        self.first_threshold=abs(max(np.max(self.occ_energies[0]),np.max(self.occ_energies[-1]))) #: float: lowest threshold of the excitations. All excitations are discrete below this level

    def determine_occ_energies(self,evals):
        """
        Extract the occupied energy levels from a Logfile BandArray structure, provided the 
        tuple of the number of occupied states
        
        Args:
            evals (BandArray): the eigenvalues as they are provided (for instance) from a `Logfile` class instance.
        """
        norb_tot=evals.info
        norb_occ=self.cm.norb_occ
        occ_energies=[]
        istart=0
        for ispin in range(len(norb_occ)):
            occ_energies.append(np.array(evals[0][istart:istart+norb_occ[ispin]]))
            istart+=norb_tot[ispin]
        self.occ_energies=occ_energies
    
    def plot_alpha(self,**kwargs):
        """Plot the imaginary part.

        Plot the real or imaginary part of the dynamical polarizability.
        
        Keyword Arguments:
           real (bool): True if real part has to be plotted. The imaginary part is plotted otherwise
           eta (float): Value of the complex imaginary part. Defaults to 1.e-2. 
           group (str): May be ``"all"``, ``"bt"``, ``"at"``;
               * ``"all"`` : provides the entire set of excitations. Default value
               * ``"bt"`` : provides only the excitations below threshold
               * ``"at"`` : provides only the excitations above threshold
           **kwargs: other arguments that might be passed to the ``meth:plot`` method of the ``mod:matplotlib.pyplot`` module.
           
        Returns:
            the reference to ``mod:matplotlib.pyplot`` module.
        """
        import matplotlib.pyplot as plt
        from futile.Utils import kw_pop
        emax=np.max(np.sqrt(self.eigenvalues))*HaeV
        kwargs,real=kw_pop('real',False,**kwargs)
        plt.xlim(xmax=emax)
        if real:
            plt.ylabel(r'$\mathrm{Re} \alpha$ (AU)',size = 14)
        else:
            plt.ylabel(r'$\mathrm{Im} \alpha$',size = 14)
            plt.yticks([])
        plt.xlabel(r'$\omega$ (eV)',size = 14)
        if hasattr(self,'first_threshold'):
            eps_h = self.first_threshold*HaeV
            plt.axvline(x=eps_h, color='black', linestyle='--')
        kwargs,eta=kw_pop('eta',1.e-2,**kwargs)
        omega=np.linspace(0.0,emax,5000)+2.0*eta*1j
        kwargs,group=kw_pop('group','all',**kwargs)
        tosp=self.avg_os
        slice=None
        if group == 'bt': slice=self.excitations_below_threshold
        if group == 'at': slice=self.excitations_above_threshold
        spectrum=self.spectrum_curves(omega,slice=slice)
        toplt=spectrum.real if real else spectrum.imag
        pltkwargs=dict(c='black',linewidth=1.5)
        pltkwargs.update(kwargs)
        plt.plot(omega*HaeV,toplt,**pltkwargs)
        return plt         
            
    def plot_excitation_landscape(self,**kwargs):
        """
        Represent the excitation landscape as splitted in the excitations class

        Args:
            **kwargs: keyword arguments to be passed to the `pyplot` instance. The ``xlabel``, ``ylabel`` as well as ``xlim`` are already set.
            
        Returns:
            reference to ``mod:matplotlib.pyplot`` module.

        Example:
           >>> ex=Excitations(cm,tm)
           >>> ex.split_excitations(evals=...,tol=1.e-4,nexc=...)
           >>> ex.plot_excitation_landscape(title='Excitation landscape')
        """
        import matplotlib.pyplot as plt
        Emin=0.0
        Emax=np.max(np.sqrt(self.eigenvalues))*HaeV
        for level in self.occ_energies[0]:
            eng_th = level*HaeV
            plt.plot((Emin,eng_th),(level,level),'--',c='red',linewidth=1)
            plt.plot((eng_th,Emax),(level,level),'-',c='red',linewidth=1)
            plt.scatter(abs(eng_th),level,marker='x',c='red')
        ind_bt=self.excitations_below_threshold
        exc_bt=np.sqrt(self.eigenvalues)[ind_bt]
        lev_bt=self.threshold_energies[ind_bt]
        plt.scatter(HaeV*exc_bt,lev_bt,s=16,marker='o',c='black')
        ind_at=self.excitations_above_threshold
        exc_at=np.sqrt(self.eigenvalues)[ind_at]
        lev_at=self.threshold_energies[ind_at]
        plt.scatter(HaeV*exc_at,lev_at,s=14,marker='s',c='blue')
        plt.xlabel('energy (eV)')
        plt.ylabel('Threshold energy (Ha)')
        plt.xlim(xmin=Emin-1,xmax=Emax)
        for attr,val in kwargs.items():
            if type(val) == dict:
                getattr(plt,attr)(**val)
            else:
                getattr(plt,attr)(val)
        return plt
    
    def dos_dict(self,group='all'):
        """Dictionary for DoS creation.
        
        Creates the keyword arguments that have to be passed to the `meth:BigDFT.DoS.append` method of the `DoS` class
        
        Args:
           group (str): May be ``"all"``, ``"bt"``, ``"at"``;
               * ``"all"`` : provides the entire set of excitations
               * ``"bt"`` : provides only the excitations below threshold
               * ``"at"`` : provides only the excitations above threshold
               
        Returns:
            dict: kewyord arguments that can be passed to the `meth:BigDFT.DoS.append` method of the `DoS` class
        
        """
        ev=np.sqrt(self.eigenvalues)
        if group == 'bt': ev=ev[self.excitations_below_threshold]
        if group == 'at': ev=ev[self.excitations_above_threshold]
        
        return dict(energies=np.array([np.ravel(ev)]),units='AU')
    
    def dos(self,group='all',**kwargs):
        """Density of States of the Excitations.
        
        Provides an instance of the `class:BigDFT.DoS.DoS` class, corresponding to the Excitations instance.
        
        Args:
           group (str): May be ``"all"``, ``"bt"``, ``"at"``;
               * ``"all"`` : provides the entire set of excitations
               * ``"bt"`` : provides only the excitations below threshold
               * ``"at"`` : provides only the excitations above threshold
            **kwargs: other arguments that might be passed to the `class:BigDFt.DoS.DoS` instantiation
               
        Returns:
            DoS: instance of the Density of States class
        """
        from BigDFT.DoS import DoS
        kwa=self.dos_dict(group=group)
        kwa['energies']=kwa['energies'][0]
        if hasattr(self,'first_threshold'): kwa['fermi_level']=self.first_threshold
        kwa.update(kwargs)
        return DoS(**kwa)
        
    def plot_Sminustwo(self,coord,alpha_ref=None,group='all'):
        """Inspect S-2 sum rule.
        
        Provides an handle to the plotting of the $S^{-2}$ sum rule, which should provide reference values for the static polarizability tensor.
        
        Args:
            coord (str): the coordinate used for inspection. May be ``'x'``, ``'y'`` or ``'z'``.
            alpha_ref (list): diagonal of the reference static polarizability tensor (for instance calculated via finite differences).
               If present the repartition of the contribution of the various groups of excitations is plotted.
            group (str): May be ``"all"``, ``"bt"``, ``"at"``;
               * ``"all"`` : provides the entire set of excitations
               * ``"bt"`` : provides only the excitations below threshold
               * ``"at"`` : provides only the excitations above threshold
        
        Returns:
            reference to ``mod:matplotlib.pyplot`` module.
        
        """
        import matplotlib.pyplot as plt
        idir=['x','y','z'].index(coord)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('energy (eV)',size = 14)
        plt.ylabel(r'$\alpha_{'+coord+coord+r'}$ (AU)',size = 14)
        if alpha_ref is not None:
            plt.axhline(y=alpha_ref[idir], color='r', linestyle='--')
        if hasattr(self,'first_threshold'):
            eps_h = abs(HaeV*self.first_threshold)
            plt.axvline(x=eps_h, color='black', linestyle='--')
        e=np.sqrt(self.eigenvalues)*HaeV
        w_ii=self.alpha_prime[:,idir]
        if group=='bt': 
            e=e[self.excitations_below_threshold]
            w_ii=w_ii[self.excitations_below_threshold]
        if group=='at':
            e=e[self.excitations_above_threshold]
            w_ii=w_ii[self.excitations_above_threshold]            
        ax1.plot(e,np.cumsum(w_ii))
        ax2 = ax1.twinx()
        ax2.plot(e,w_ii,color='grey',linestyle='-')
        plt.ylabel(r'$w_{'+coord+coord+r'}$ (AU)',size=14)
        
        return plt

                        
def build_eigenp_dict(numOrb,nvirt,Cbig,Dbig,na):
    """
    Build the dictionary with the solutions of the eigenproblems for each choice of na
    
    """
    C_ext,dipoles_ext=extract_subset([numOrb],[nvirt],Cbig,Dbig,[na])
    E2,C_E2 = np.linalg.eigh(C_ext)
    C_E2 = C_E2.T
    newdipole=[]
    for line in dipoles_ext:
        newdipole.append(line[0]*np.array(line[1:]))
    newdipole=np.array(newdipole)
    write('Eigensystem solved for',na)
    return {'Cmat':C_ext,'eigenvalues':E2,'eigenvectors':C_E2,'transitions':newdipole}

def solveEigenProblems(numOrb,box,nalpha):
    eigenproblems = {}
    for na in nalpha:
        if na > box['nvirt']:
            print 'There are not enough virtual states for', na
            continue
        eigenproblems[na]=build_eigenp_dict(numOrb,box['nvirt'],box['C'],box['T'],na)
    return eigenproblems

def diagonalize_CM(norb,syst,naSmall):
    for rVal,box in syst.iteritems():
        write('Solve for rVal = ', rVal)   
        ep = solveEigenProblems(norb,box,naSmall)
        box['eigenproblems'] = ep

def get_alpha_energy(log,norb,nalpha):
    return log.evals[0][0][norb+nalpha-1]

def get_p_energy(log,norb):
    return log.evals[0][0][0:norb]


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
    """
    Collect the spectrum in all the box using the highest number of virtual orbitals
    """
    sp = {}
    for rVal in syst:
        nalpha = syst[rVal]['eigenproblems'].keys()
        nalpha.sort()
        nvirt = nalpha[-1]
        print 'Compute for rVal = ', rVal,' with nalpha = ', nvirt
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
    pProj = pProj[0:numOrb]+pProj[numOrb:2*numOrb] # halves the components 
    return pProj

def get_p_energy(log,norb):
    return log.evals[0][0][0:norb]

def get_threshold(pProj,th_energies,th_levels,tol):
    norb=len(th_energies)
    pProj = pProj.tolist()
    pProj.reverse()
    imax=norb-1
    for val in pProj:
        if val > tol: break
        imax-=1
    return [th_levels[imax],th_energies[imax]]

def find_excitation_thr(dict_casida,na,nexc,th_energies,th_levels,tol):
    norb=len(th_energies)
    thrs=[]
    for a in range(nexc):
        proj=identify_contributions(norb,na,a,dict_casida['eigenvectors'])
        th=get_threshold(proj,th_energies,th_levels,tol)
        thrs.append(th)
    dict_casida['thresholds']=np.array(thrs)

def collect_excitation_thr(syst,numOrb,numExc,th_levels,tol):
    for rVal in syst:
        nalpha = syst[rVal]['eigenproblems'].keys()
        nalpha.sort()
        nvirt = nalpha[-1]
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        th_energies = HaeV*abs(get_p_energy(syst[rVal]['logfile'],numOrb))
        find_excitation_thr(dict_casida,nvirt,numExc,th_energies,th_levels,tol)  

def build_index_bt_at(dict_casida):
    index_bt = []
    index_at = []
    numExc = len(dict_casida['thresholds'])
    for ind in range(numExc):
        if HaeV*np.sqrt(dict_casida['eigenvalues'][ind]) < dict_casida['thresholds'][ind][1]:
            index_bt.append(ind)
        else : index_at.append(ind)
    return index_bt,index_at

def collect_index_bt_at(syst):
    index_bt = {}
    index_at = {}
    for rVal in syst:
        nalpha = syst[rVal]['eigenproblems'].keys()
        nalpha.sort()
        nvirt = nalpha[-1]   
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        ind_bt,ind_at = build_index_bt_at(dict_casida)
        index_bt[rVal] = ind_bt
        index_at[rVal] = ind_at
    return index_bt,index_at

def identify_channels(dict_casida,numOrb,numExc,th_levels):
    chn = {}
    for lev in th_levels:
        chn[lev] = []
    for exc in range(numExc):
        lev = dict_casida['thresholds'][exc][0]
        chn[lev].append([exc,HaeV*np.sqrt(dict_casida['eigenvalues'][exc])])
    return chn  

def collect_channels(syst,numOrb,numExc,th_levels):
    channels = {}
    for rVal in syst:
        nalpha = syst[rVal]['eigenproblems'].keys()
        nalpha.sort()
        nvirt = nalpha[-1]   
        dict_casida = syst[rVal]['eigenproblems'][nvirt]
        channels[rVal] = identify_channels(dict_casida,numOrb,numExc,th_levels)
    return channels





