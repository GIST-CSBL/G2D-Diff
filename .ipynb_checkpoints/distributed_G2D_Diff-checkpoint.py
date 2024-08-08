import torch
from torch import nn 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils import shuffle
import copy
import accelerate
from accelerate import Accelerator
import pickle

from src.utils.g2d_diff_genodrug_dataset import *
from src.g2d_diff_ce import *
from src.g2d_diff_diff import *


from einops import rearrange, repeat, reduce
from functools import partial
import math
import os
import torch.distributed as dist



def main():
    
    ##############
    # Data load
    ##############
    PREDIFINED_GENOTYPES = ['mut', 'cna', 'cnd']


    nci_data = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
    nci_data = nci_data.dropna()

    val_cell = ['EKVX_LUNG', 'SKMEL28_SKIN', 'SKOV3_OVARY', 'NCIH226_LUNG', 'OVCAR4_OVARY']
    test_cell = ['TK10_KIDNEY', 'OVCAR5_OVARY', 'HOP92_LUNG', 'SKMEL2_SKIN', 'HS578T_BREAST']

    nci_data_train = nci_data[~nci_data['ccle_name'].isin(val_cell + test_cell)]



    

    cell2mut = pd.read_csv("./data/drug_response_data/original_cell2mut.csv", index_col = 0).rename(columns={'index':'ccle_name'})
    cell2cna = pd.read_csv("./data/drug_response_data/original_cell2cna.csv", index_col = 0).rename(columns={'index':'ccle_name'})
    cell2cnd = pd.read_csv("./data/drug_response_data/original_cell2cnd.csv", index_col = 0).rename(columns={'index':'ccle_name'})
    drug2smi = pd.read_csv("./data/drug_response_data/DC_drug2smi.csv").iloc[:, 0:-1]


    dataset_obj = GenoDrugDataset(nci_data_train, cell2mut, drug2smi, cna=cell2cna, cnd=cell2cnd)
    collate_fn = GenoDrugCollator(genotypes=PREDIFINED_GENOTYPES)

    class_count = []
    for i in range(5):
        class_count.append(len(nci_data_train[nci_data_train['auc_label']==i]))
    class_count = np.array(class_count)
    weight = 1. / class_count
    samples_weight = np.array([weight[t] for t in nci_data_train['auc_label']])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    ##############
    # Model load
    ##############
  
    accelerate.utils.set_seed(42)

    ## Change here to change batchsize (means batch size for each GPU. If 4 gpu, batch size is 128 * 4 = 512)
    batch_size = 128
    max_epochs = 5000
    
    accelerator = Accelerator()    
    device = accelerator.device
 
   
    diff_model = Diffusion(device = device, training=True, prand = 0.1).to(device).to(torch.float)
    
   
    optimizer = optim.Adam([p for p in diff_model.parameters() if p.requires_grad == True], lr = 1e-4)
    tr_loader = DataLoader(dataset_obj, batch_size=batch_size, drop_last=True, collate_fn=collate_fn, sampler = sampler)

    diff_model, optimizer, tr_loader = accelerator.prepare(diff_model, optimizer, tr_loader)
    
    total_loss = []
    

    ##############
    # Training
    ##############

    for epoch in range(max_epochs):
        epoch_loss = []


        for i, batch in tqdm(enumerate(tr_loader), total = len(tr_loader)):
            ## Batch data load to device
            for key in batch.keys():
                if 'genotype' in key:
                    for mut in batch[key].keys():
                        batch[key][mut] = batch[key][mut].to(device)
                elif key == 'cell_name':
                    None
                elif key == 'drug_name':
                    None
                else:
                    batch[key] = batch[key].to(device)



            loss = diff_model(batch)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()


            epoch_loss.append(loss.detach().item())

            
        
        
        
        accelerator.wait_for_everyone()

        total_loss.append(np.mean(epoch_loss))
        print("Epoch: ", epoch, " Loss: ", np.mean(epoch_loss))

        unwrapped_model = accelerator.unwrap_model(diff_model)

        # Use here to save
#         accelerator.save({
#                 'diffusion_state_dict': unwrapped_model.state_dict(),
#                 'solver_state_dict': optimizer.state_dict(),
#                 'loss_traj': total_loss
#             }, "diffusion_models/1229_512_adanorm_6layers_%d.ckpt"%(epoch))
        
       
       



if __name__ == "__main__":
    main()   