

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import itertools



class GenoDataset(Dataset):

    def __init__(self, response_data, cell2mut, **kwargs):

        self.input_df = response_data 
        self.cell2mut = cell2mut.set_index(['ccle_name'])
  
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
        cell, res_class = self.input_df.iloc[index].values
        
        cell_genotype = {}
        cell_genotype['mut'] = torch.FloatTensor(self.cell2mut.loc[cell].values)
        if self.cell2cna is not None:
            cell_genotype['cna'] = torch.FloatTensor(self.cell2cna.loc[cell].values)
        if self.cell2cnd is not None:
            cell_genotype['cnd'] = torch.FloatTensor(self.cell2cnd.loc[cell].values)    
        
            
        result_dict = dict()
        result_dict['genotype'] = cell_genotype
        result_dict['class'] = res_class
        result_dict['cell_name'] = cell
        
        return result_dict

                                                                             
class GenoCollator(object):
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
        result_dict['genotype'] = mutation_dict
        result_dict['class'] = torch.LongTensor([dr['class'] for dr in data])

        return result_dict
