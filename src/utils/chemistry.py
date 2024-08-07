from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


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