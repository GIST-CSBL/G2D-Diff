'''
    "analysis" module with multi-processing capability
'''

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from vae_package import pack_global
from vae_package.multiprocess_tools import multiprocess_task_on_list, multiprocess_task_many_args

# import same names from tools for having the same functions
from vae_package.analysis import convert_to_canon
from vae_package.analysis import evaluation_basic
from vae_package.analysis import calculate_simmat
from vae_package.analysis import internal_diversity

def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol == None: return False
    return True

def get_valid_canons(smilist):
    '''
        Get the valid & canonical form of the smiles.
        Please note that different RDKit version could result in different validity for the same SMILES.
    '''
    canons = multiprocess_task_on_list(convert_to_canon, smilist, pack_global.NJOBS_MULTIPROC)
    canons = np.array(canons)
    invalid_ids = np.where(canons==None)[0]
    # insert error string to invalid positions
    canons[invalid_ids] = "<ERR>"

    # Re-checking the parsed smiles, since there are bugs in rdkit parser.
    # https://github.com/rdkit/rdkit/issues/4701
    is_valid = multiprocess_task_on_list(is_valid_smiles, canons, pack_global.NJOBS_MULTIPROC)
    is_valid = np.array(is_valid)
    invalid_ids = np.where(is_valid==False)[0]
    return np.delete(canons, invalid_ids), invalid_ids

def get_morganfp_by_smi(smi, r=2, b=2048):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=b)
    return fp

def get_fps_from_smilist(smilist, r=2, b=2048):
    """ We assume that all smiles are valid. """
    # zipped input format
    _r = [r for _ in range(len(smilist))]
    _b = [b for _ in range(len(smilist))]
    zipped_input = zip(smilist, _r, _b)
    fps_list = multiprocess_task_many_args(get_morganfp_by_smi, zipped_input, pack_global.NJOBS_MULTIPROC)
    return fps_list

def fps_to_npfps(fps_list):
    npfps_list = multiprocess_task_on_list(np.array, fps_list, pack_global.NJOBS_MULTIPROC)
    return np.array(npfps_list)

def npfp2rdkfp(npfp):
    bitstring="".join(npfp.astype(str))
    return DataStructs.cDataStructs.CreateFromBitString(bitstring)

def npfps_to_rdkfps(npfps):
    rdkfps = multiprocess_task_on_list(npfp2rdkfp, npfps, pack_global.NJOBS_MULTIPROC)
    return rdkfps

