

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import itertools
from . import smiles_vocab


class CandidateDataset(Dataset):

    def __init__(self, response_data, cell2mut, drug2smi, sca2smi, **kwargs):

        self.input_df = response_data 
        self.cell2mut = cell2mut.set_index(['ccle_name'])
        self.drug2smi = drug2smi.set_index(['drug'])
        self.sca2smi = sca2smi.set_index(['drug'])
        
        self.cell2cna, self.cell2cnd = None, None
        if kwargs is not None:
            for key, item in kwargs.items():
                if key == 'cna':
                    self.cell2cna = item.set_index(['ccle_name'])
                elif key == 'cnd':
                    self.cell2cnd = item.set_index(['ccle_name'])
            

    def __len__(self):
        return self.input_df.shape[0]

    def __getitem__(self, index):
        cell, drug, auc, res_class = self.input_df.iloc[index].values
        
        cell_genotype = {}
        cell_genotype['mut'] = torch.FloatTensor(self.cell2mut.loc[cell].values)
        if self.cell2cna is not None:
            cell_genotype['cna'] = torch.FloatTensor(self.cell2cna.loc[cell].values)
        if self.cell2cnd is not None:
            cell_genotype['cnd'] = torch.FloatTensor(self.cell2cnd.loc[cell].values)    
        
    
        drug_latent_vec = torch.FloatTensor(self.drug2smi.loc[drug].iloc[1:].values.astype('float32'))
        sca_latent_vec = torch.FloatTensor(self.sca2smi.loc[drug].iloc[1:].values.astype('float32'))
            
        result_dict = dict()
        result_dict['genotype'] = cell_genotype
        result_dict['drug'] = drug_latent_vec
        result_dict['scaffold'] = sca_latent_vec
        result_dict['class'] = res_class
        result_dict['cell_name'] = cell
        result_dict['drug_name'] = drug
        return result_dict

                                                                             
class CandidateCollator(object):
    def __init__(self, genotypes):
        """
        Collator for data
        """
        
        self.genotypes = genotypes

    def __call__(self, data):
        result_dict = dict()
        mutation_dict = dict()

        for genotype in self.genotypes:
            mutation_dict[genotype] = torch.stack([dr['genotype'][genotype] for dr in data])
        
        
        result_dict['cell_name'] = [dr['cell_name'] for dr in data]
        result_dict['drug_name'] = [dr['drug_name'] for dr in data]
        result_dict['genotype'] = mutation_dict
        result_dict['drug'] = torch.stack([dr['drug'] for dr in data])
        result_dict['scaffold'] = torch.stack([dr['scaffold'] for dr in data])
        result_dict['class'] = torch.LongTensor([dr['class'] for dr in data])

        return result_dict

# class CandidateDataset(Dataset):

#     def __init__(self, response_data, cell2mut, drug2smi, **kwargs):

#         self.input_df = response_data #this is dataframe because dataset is split to train and val
#         # the other will be read from csv
#         self.cell2mut = cell2mut.set_index(['ccle_name'])
#         self.drug2smi = drug2smi.set_index(['drug'])
        
#         self.cell2cna, self.cell2cnd = None, None
#         if kwargs is not None:
#             for key, item in kwargs.items():
#                 if key == 'cna':
#                     self.cell2cna = item.set_index(['ccle_name'])
#                 elif key == 'cnd':
#                     self.cell2cnd = item.set_index(['ccle_name'])
            

#     def __len__(self):
#         return self.input_df.shape[0]

#     def __getitem__(self, index):
#         cell, drug, auc, res_class = self.input_df.iloc[index].values
        
#         cell_genotype = {}
#         cell_genotype['mut'] = self.cell2mut.loc[cell].values
#         if self.cell2cna is not None:
#             cell_genotype['cna'] = self.cell2cna.loc[cell].values
#         if self.cell2cnd is not None:
#             cell_genotype['cnd'] = self.cell2cnd.loc[cell].values
        
    
#         smiles = list(self.drug2smi.loc[drug].values)
            
#         result_dict = dict()
#         result_dict['genotype'] = cell_genotype
#         result_dict['drug'] = smiles
#         result_dict['class'] = res_class
#         result_dict['cell_name'] = cell
#         return result_dict

    
    
# class CandidateCollator(object):
#     def __init__(self, genotypes, token_path):
#         """
#         Collator for data
#         """
        
#         self.genotypes = genotypes
#         vocab = smiles_vocab.Vocabulary()
#         vocab.init_from_file(token_path)
        
        
#         self.tokenizer = smiles_vocab.SmilesTokenizer(vocab)
#         self.vocab_obj = vocab

    

#     # The use of collate_fn is for (<PAD>)-padding. 
#     # The arr is expected to be an encoded smiles array;
#     # that is, each element is float.
#     def collate_fn(self, arr, PAD_idx, BEG_idx):
#         """Function to take a list of encoded sequences and turn them into a batch"""
#         max_length = max([seq.size for seq in arr])+1
#         collated_arr = torch.full((len(arr), max_length), PAD_idx, dtype=torch.float32)
#         for i, seq in enumerate(arr):
#             collated_arr[i, 0] = BEG_idx
#             collated_arr[i, 1:seq.size+1] = torch.Tensor(seq)
#         return collated_arr

#     def prepare_batch(self, smiles_list):
#         """ 
#         Get a batch of SMILES, turn them into a training batch.
#         Also, return the index of the smiles that raised KeyError during encoding.
#         Thus, the size of smiles_list is not always equal to sample_batch,
#         since error-occurring smiles are removed.
#         """
#         EOS_idx = self.vocab_obj.get_EOS_idx()
#         PAD_idx = self.vocab_obj.get_PAD_idx()
#         BEG_idx = self.vocab_obj.get_BEG_idx()
#         sample_batch_t = [self.tokenizer.tokenize(smiles) for smiles in smiles_list]
#         sample_batch_e = []
#         keyerror_ids = []
#         for i, tokens in enumerate(sample_batch_t):
#             try:
#                 encoded = self.vocab_obj.encode(tokens)
#             except KeyError as err:
#                 keyerror_ids.append(i)
#                 print("KeyError at %s"%smiles_list[i])
#                 continue
#             sample_batch_e.append(encoded)
#       # add <EOS> at the end
#         EOS_batch = []
#         for tokens in sample_batch_e:
#             tokens = list(tokens)
#             tokens.append(EOS_idx)
#             EOS_batch.append(np.array(tokens, dtype=np.float32))
#         # pad each example to the length of the longest in the batch
#         sample_batch = self.collate_fn(EOS_batch, PAD_idx, BEG_idx)
#         return sample_batch, keyerror_ids


#     def __call__(self, data):
#         result_dict = dict()
#         mutation_dict = dict()

#         for genotype in self.genotypes:
#             mutation_dict[genotype] = [dr['genotype'][genotype] for dr in data]
        

#         seqs, _ = self.prepare_batch(list(itertools.chain(*[dr['drug'] for dr in data])))
        
#         result_dict['cell_name'] = [dr['cell_name'] for dr in data]
#         result_dict['genotype'] = mutation_dict
#         result_dict['drug'] = seqs
#         result_dict['class'] = [dr['class'] for dr in data]

#         return result_dict


