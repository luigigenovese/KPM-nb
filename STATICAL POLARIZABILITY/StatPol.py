from BigDFT import Datasets as D, Calculators as C, Inputfiles as I, Logfiles as lf
from futile.Utils import write
import numpy as np
import os

def get_molecule_database():
    """
    Scan the working directory and store in a list the name of folder.
    During the scan the file and the "service" directories are neglected

    Todo : improve this method when the flow for the construction of the GS
    will be defined
    """
    mol_database = os.listdir('.')
    for f in reversed(mol_database):
        if os.path.isdir(f) == False:
            mol_database.remove(f)
    if os.path.isdir('.ipynb_checkpoints'):
        mol_database.remove('.ipynb_checkpoints')
    return mol_database

def build_alpha_dataset(**kwargs):
    """
    Create the dataset and append the runs needed to compute the statical polarizability
    for a specific choice of the input parameters. Set also a postprocessing function
    to extract the value of alpha.

    Args:
        kwargs['intensity'] : the intensity of the field
        kwargs['input']     : the input file
        kwargs['posinp']    : the posinp
        kwargs['ppf']       : the postprocessing function
        kwargs['runner']    : the instance of SystemCalculator
    """
    lbl = 'alpha_'+str(kwargs['intensity'])
    study = D.Dataset(label=lbl,run_dir='alpha',intensity=kwargs['intensity'],posinp=kwargs['posinp'])
    study.set_postprocessing_function(kwargs['ppf'])

    f = kwargs['intensity']
    inp = kwargs['input']
    for ind,sign in enumerate(['+','-']):
        for idir,coord in enumerate(['x','y','z']):
            el=np.zeros(3)
            el[idir]=(1-2*ind)*f
            inp.apply_electric_field(el.tolist())
            idd = {'rmult':inp['dft']['rmult'][0],'dir':coord,'sign':sign,'F':f}
            study.append_run(id=idd,runner=kwargs['runner'],input=inp)

    return study

def eval_alpha(study):
    """"
    Extract the statical polarizability tensor from the study dataset.
    Assumes that the run of the dataset have been appended by build_alpha_dataset
    """
    dipoles = study.fetch_results(attribute = 'dipole')
    f = study.get_global_option('intensity')
    alpha=np.mat(np.zeros(9)).reshape(3,3)
    for ind in range(3):
        alpha[ind] = np.array(dipoles[ind])-np.array(dipoles[ind+3])
    alpha = alpha.T / (2.0*f)
    return alpha

def eval_alpha_avg(a):
    """
    Return the average diagonal component of the polarizability tensor
    """
    avg = a.trace()/3.
    return avg[0,0]

def iterate_parameter(**kwargs):
    """
    Perform the iteration over the values of a parameter and compute the polarizability
    tensor. Return a dictionary with the same structure of the output of seek_convergence
    Args:
        kwargs['label']     : the name of the parameter
        kwargs['values']    : the array with the values of the parameter
        kwargs['data']      : the array with the dataset buit with kwargs['values']
    """

    label = kwargs['label']
    values = kwargs['values']
    data = kwargs['data']
    results = {}
    for ind,v in enumerate(values):
        print 'Run the dataset with', label, v
        results[v] = data[ind].run()

    out = {'label':label,'values':values,'results':results,'converged':None,'converged_values':None}

    return out

def seek_convergence(at=1e-3,rt=1e-2,**kwargs):
    """
    Perform a convergence procedure by using 3 values of a parameter.
    Return a dictionary with the input parameters, the results of all
    the computation performed, the value of the convergence parameter and a
    boolean that states if the convergence procedure succeeds or not

    Args:
        kwargs['label']     : the name of the convergence parameter
        kwargs['values']    : the array with the 3 ordered values of the convergence parameter
        kwargs['data']      : the array with the dataset buit with kwargs['values']
        at,rt               : absolute and relative tol of np.allclose
    """
    label = kwargs['label']
    values = kwargs['values']
    data = kwargs['data']
    results = {}
    for ind,v in enumerate(values[:2]):
        print 'Run the dataset with', label, v
        results[v] = data[ind].run()

    out = {'label':label,'values':values}

    convergence = np.allclose(results[values[0]],results[values[1]],atol = at, rtol = rt)
    if convergence:
        write('Convergence achieved for', label ,values[0])
        results[values[2]] = None
        out['results'] = results
        out['converged'] = True
        out['converged_value'] = values[0]
    else:
        write('Convergence for', label,values[0],'failed')
        write('Set the value of the parameter to', values[1])
        write('')
        print 'Run the dataset with', label, values[2]
        results[values[2]]=data[2].run()
        out['results'] = results
        convergence = np.allclose(results[values[1]],results[values[2]],atol = at, rtol = rt)
        if convergence:
            write('Convergence achieved for', label ,values[1])
            out['converged'] = True
            out['converged_value'] = values[1]
        else:
            write('Convergence for', label,values[1],'failed')
            write('Return the value associated to',label,values[2],'. Perform further check!!!')
            out['converged'] = False
            out['converged_value'] = values[2]
    return out

def perform_field_iteration(**kwargs):
    """
    Perform the iteration w.r.t. the intensity of the static field to extract the
    result of the polarizability tensor.

    Args:
        kwargs['field_int'] : list with the intensity of the field
        kwargs['input']     : the input file
        kwargs['posinp']    : the posinp
        kwargs['ppf']       : the postprocessing function
        kwargs['runner']    : the instance of SystemCalculator
    """
    code=kwargs['runner']
    field_int=kwargs['field_int']
    code.update_global_options(verbose=False)

    data = []
    for f in field_int:
        data.append(build_alpha_dataset(intensity=f,input=kwargs['input'],runner=code,posinp=kwargs['posinp'],ppf=kwargs['ppf']))
    out = iterate_parameter(label='field_int',values=field_int,data=data)
    return out

def perform_field_convergence(at=1e-3,rt=1e-2,field_int=[1e-2,5e-3,1e-3],**kwargs):
    """
    Perform the convergence procedure w.r.t. the intensity of the static field to extract the
    result of the polarizability tensor.

    Args:
        kwargs['input']     : the input file
        kwargs['posinp']    : the posinp
        kwargs['ppf']       : the postprocessing function
        kwargs['runner']    : the instance of SystemCalculator
        at,rt               : absolute and relative tol of np.allclose
    """
    code=kwargs['runner']
    code.update_global_options(verbose=False)

    data = []
    for f in field_int:
        data.append(build_alpha_dataset(intensity=f,input=kwargs['input'],runner=code,posinp=kwargs['posinp'],ppf=kwargs['ppf']))
    out = seek_convergence(rt=rt,label='field_int',values=field_int,data=data)
    return out

def build_rmult_list(gs):
    """
    Return a set of values of rmult. The set starts at the value of rmult of the gs
    calculation and contains 3 values to perform a convergence procedure analogous
    to the one performed w.r.t. the field intensity
    """

    r0 = gs.log['dft']['rmult']
    rmult_list = []
    for incr in range(3):
        rmult_list.append([r0[0]+incr,r0[1]])
    return rmult_list

def perform_rmult_iteration(**kwargs):
    """
    Perform the iteration w.r.t. the coarse value of rmult to extract the
    result of the polarizability tensor.

    Args:
        kwargs['rmult']     : list of values of rmult in the form [coarse,fine]
        kwargs['intensity'] : intensity of the field
        kwargs['input']     : the input file
        kwargs['posinp']    : the posinp
        kwargs['ppf']       : the postprocessing function
        kwargs['runner']    : the instance of SystemCalculator
    """
    rmult=kwargs['rmult']
    f=kwargs['intensity']
    inp=kwargs['input']
    code=kwargs['runner']
    code.update_global_options(verbose=False)

    data = []
    coarse = []
    for r in rmult:
        coarse.append(r[0])
        inp.set_rmult(r)
        data.append(build_alpha_dataset(intensity=f,input=inp,runner=code,posinp=kwargs['posinp'],ppf=kwargs['ppf']))
    out = iterate_parameter(label='rmult',values=coarse,data=data)
    return out

def perform_rmult_convergence(at=1e-3,rt=1e-2,**kwargs):
    """
    Perform the convergence procedure w.r.t. the coarse value of rmult to extract the
    result of the polarizability tensor.

    Args:
        kwargs['rmult']     : list of values of rmult in the form [coarse,fine] (3 values for coarse)
        kwargs['intensity'] : intensity of the field
        kwargs['input']     : the input file
        kwargs['posinp']    : the posinp
        kwargs['ppf']       : the postprocessing function
        kwargs['runner']    : the instance of SystemCalculator
        at,rt               : absolute and relative tol of np.allclose
    """
    rmult=kwargs['rmult']
    f=kwargs['intensity']
    inp=kwargs['input']
    code=kwargs['runner']
    code.update_global_options(verbose=False)

    data = []
    coarse = []
    for r in rmult:
        coarse.append(r[0])
        inp.set_rmult(r)
        data.append(build_alpha_dataset(intensity=f,input=inp,runner=code,posinp=kwargs['posinp'],ppf=kwargs['ppf']))
    out = seek_convergence(rt=rt,label='rmult',values=coarse,data=data)
    return out
