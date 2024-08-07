from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.decomposition import PCA
import sascorer as sas
from rdkit.Chem import QED

# Max size for single OT calculation
MAX_MOLS_OT = 500
# conversion of Tanimoto similarity to the distance
tansim_to_dist = lambda ts: np.power(10, 1-ts) - 1
def convert_to_canon(smi, verbose=None):
    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        if verbose: print('[ERROR] cannot parse: ', smi)
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def get_valid_canons(smilist):
    '''
        Get the valid & canonical form of the smiles.
        Please note that different RDKit version could result in different validity for the same SMILES.
    '''
    canons = []
    invalid_ids = []
    for i, smi in enumerate(smilist):
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            invalid_ids.append(i)
            canons.append(None)
        else:
            canons.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    # Re-checking the parsed smiles, since there are bugs in rdkit parser.
    # https://github.com/rdkit/rdkit/issues/4701
    re_canons = []
    for i, smi in enumerate(canons):
        if smi == None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            print("rdkit bug occurred!!")
            invalid_ids.append(i)
        else:
            re_canons.append(smi)
    return re_canons, invalid_ids

def get_morganfp_by_smi(smi, r=2, b=2048):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=b)
    return fp

def get_fps_from_smilist(smilist, r=2, b=2048):
    """ We assume that all smiles are valid. """
    fps = []
    for i, smi in enumerate(smilist):
        fps.append(get_morganfp_by_smi(smi, r, b))
    return fps

def fps_to_npfps(fps_list):
    """ fps_list: list of MorganFingerprint objects """
    return np.array(fps_list)

def npfps_to_rdkfps(npfps):
    rdkfps = []
    for npfp in npfps:
        bitstring="".join(npfp.astype(str))
        rdkfp = DataStructs.cDataStructs.CreateFromBitString(bitstring)
        rdkfps.append(rdkfp)
    return rdkfps

def evaluation_basic(sample_size, vacans, pretrainset):
    """
        evaluate Validity, Uniqueness, Novelty
        - vacans: list of valid & canonical smiles
        - sample_size: # of the generator samples
    """
    validity = len(vacans) / sample_size
    if validity <= 0:
        return 0, 0, 0
    unis = list(set(vacans))
    uniqueness = len(unis) / len(vacans)
    novs = list(set(unis).difference(set(pretrainset)))
    novelty = len(novs) / len(unis)
    return validity, uniqueness, novelty

def calculate_simmat(fps1, fps2):
    """ Calculate the similarity matrix between two fingerprint lists. """
    simmat = np.zeros((len(fps1), len(fps2)))
    for i in range(len(fps1)):
        for j in range(len(fps2)):
            simmat[i,j] = DataStructs.FingerprintSimilarity(fps1[i], fps2[j])
    return simmat

def internal_diversity(simmat):
    return (1-simmat).mean()

def optimal_transport(smilist1, smilist2, max_mols_ot=MAX_MOLS_OT):
    """ 
        Given two sets of molecules with same size,
        compute the optimal transport mapping, 
        return mean value of (repeatition N/MAX_MOLS_OT) times OT calculations.
    """
    if len(smilist1) != len(smilist2):
        print("Please use the same size for smilist1 and smilist2")
        return None

    N = len(smilist1)
    repetitions = int(N/max_mols_ot)
    motd_list = []
    for i in range(repetitions):
        s1 = smilist1[i*max_mols_ot : (i+1)*max_mols_ot]
        s2 = smilist2[i*max_mols_ot : (i+1)*max_mols_ot]
        fps1 = get_fps_from_smilist(s1)
        fps2 = get_fps_from_smilist(s2)
        simmat = calculate_simmat(fps1, fps2)
        distmat = tansim_to_dist(simmat)    # convert simmilarity to distance
        # calculate the best transportation mapping
        row_ind, col_ind = optimize.linear_sum_assignment(distmat)
        # calculate the mean of the transportation costs
        motd = distmat[row_ind, col_ind].mean()
        motd_list.append(motd)
    return np.mean(motd_list)

# QED
def get_QEDs(mols):
    return [QED.qed(mol) for mol in mols]

# SAS
def get_SASs(mols):
    return [sas.calculateScore(mol) for mol in mols]


import fcd

def fcd_calculation(smilist1, smilist2):
    """
        calculate FC distance between two smiles sets
        smilist1, smilist2 <- list of smiles
    """
    chnt_model = fcd.load_ref_model()   # load ChemNet model
    vectors1 = fcd.get_predictions(chnt_model, smilist1)
    vectors2 = fcd.get_predictions(chnt_model, smilist2)
    mu1, sigma1 = np.mean(vectors1, axis=0), np.cov(vectors1.T)
    mu2, sigma2 = np.mean(vectors2, axis=0), np.cov(vectors2.T)
    return fcd.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)