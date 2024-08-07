

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import itertools



class GenoDrugDataset(Dataset):

    def __init__(self, response_data, cell2mut, drug2smi, **kwargs):

        self.input_df = response_data 
        self.cell2mut = cell2mut.set_index(['ccle_name'])
        self.drug2smi = drug2smi.set_index(['drug'])
        
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
            
        result_dict = dict()
        result_dict['genotype'] = cell_genotype
        result_dict['drug'] = drug_latent_vec
        result_dict['class'] = res_class
        result_dict['cell_name'] = cell
        result_dict['drug_name'] = drug
        result_dict['auc'] = auc
        return result_dict

                                                                             
class GenoDrugCollator(object):
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
        result_dict['class'] = torch.LongTensor([dr['class'] for dr in data])
        result_dict['auc'] = torch.FloatTensor([dr['auc'] for dr in data])

        return result_dict
