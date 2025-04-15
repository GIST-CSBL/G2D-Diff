import torch
import pandas as pd
import numpy as np

from torch import nn 
import random
import os


from src.g2d_diff_ce import *

class ResponsePredictor(nn.Module):
    def __init__(self,  cond_dim = 128, drug_dim = 128, cond_layers = [64,  1],  out_size = 1, device = 'cuda'):
        super(ResponsePredictor, self).__init__()

        self.cond_enc = nn.Sequential(nn.Linear(cond_dim, cond_dim), 
                                      nn.GELU(), 
                                      nn.Dropout(0.1), 
                                      nn.Linear(cond_dim, cond_dim),
                                      nn.GELU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(cond_dim, cond_dim))
        
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, drug_dim), 
                                      nn.GELU(), 
                                      nn.Dropout(0.1), 
                                      nn.Linear(drug_dim, drug_dim),
                                      nn.GELU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(drug_dim, drug_dim))
        
        self.downsampler = nn.Sequential(nn.Linear(cond_dim + drug_dim, drug_dim),
                                         nn.GELU(),
                                         nn.Dropout(0.1),
                                         nn.Linear(drug_dim, cond_layers[0]),
                                         nn.GELU(),
                                         nn.Dropout(0.1),
                                         nn.Linear(cond_layers[0], cond_layers[1]))
 

        self.drug_dim = drug_dim
        self.device_name = device


                
    
    def forward(self, inputs):
        drug, cond = inputs
        batch_size = drug.shape[0]
        
        cond_emb = self.cond_enc(cond)
        drug_emb = self.drug_enc(drug)
        
        fin_feat = torch.cat([cond_emb, drug_emb], dim = 1)
        
        out_prob = self.downsampler(fin_feat)

        
        return out_prob
    
    
class RES_Condition_Encoder(nn.Module):
    def __init__(self, num_of_genotypes, num_of_dcls, num_of_genes = 718,  gene_emb_size = 128,  device = 'cuda', neighbor_info = True):
        super(RES_Condition_Encoder, self).__init__()
        
        self.num_of_genes = num_of_genes
        self.gene_emb_size = gene_emb_size
        self.device = device
        
        self.gene_embedding = nn.Embedding(num_of_genes, gene_emb_size)
        self.dcls_embedding = nn.Embedding(num_of_dcls, gene_emb_size)
        self.muts_embedding = nn.Embedding(num_of_genotypes, gene_emb_size)
        
        self.cls_embedding = nn.Embedding(1, gene_emb_size)
        self.mut2signal = Mut2Signal(emb_size = gene_emb_size)
        self.T_neigh = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
        self.T_whole = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8)) 
        self.T_reout = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
             
        ############################################################################################################################################
        
        if neighbor_info:
            print("NeST neighbor info is used")
            with open("./data/NeST_neighbor_adj.npy", "rb") as f:
                self.gene_adj = torch.BoolTensor(np.load(f)).to(device)
        else: 
            print("No prior knowledge is used")
            self.gene_adj = None

    def get_new_adj(self, adj):
        new_adj = torch.ones(self.num_of_genes + 1, self.num_of_genes + 1).to(self.device)
        new_adj[:-1, :-1] = adj
        return new_adj
    
    def forward(self, data: Dict):

        batch_len = len(data['class'])
        _genotype = data['genotype']
        
        gene_list = np.array([x for x in range(self.num_of_genes)])
        _genes = self.gene_embedding(torch.IntTensor(gene_list).to(self.device)).unsqueeze(0).expand(batch_len,-1, -1)
        
        cls_list = np.array([0 for _ in range(batch_len)])
        _class = self.cls_embedding(torch.IntTensor(cls_list).to(self.device)).view(batch_len, 1, -1)
        revised_adj = self.get_new_adj(self.gene_adj)

        _gene_add = _genes
        whole_att_list = []
        mut_keys = list(_genotype.keys()) # ['MUT', 'CNA', 'CND']
        for i, mut_type in enumerate(mut_keys):
            # Generate the mutation embedding
            # Normalize + transform to each gene's perspective
            mut_list = np.array([i for _ in range(self.num_of_genes)])
            _mut_base = self.muts_embedding(torch.IntTensor(mut_list).to(self.device)).unsqueeze(0).expand(batch_len,-1, -1)
            mut_emb_affn = self.mut2signal(_genes, _mut_base)
            # Generate mutation mask
            mut_emb_mask = _genotype[mut_type].unsqueeze(-1).expand(-1,-1, self.gene_emb_size)
            # Add the mutation signal to the gene embedding
            _gene_add = _gene_add + (mut_emb_mask * mut_emb_affn)
            
        ########################################## PROPOGATION BEGIN ##########################################
        # Concat response class vector
        gsj_cat = torch.cat([_gene_add, _class], axis = 1)
        # Propogate through NeST siblings
        _gs, _ = self.T_neigh(x = gsj_cat, mask = revised_adj)
        #whole_att_list.append(_att.detach().cpu().numpy()) 
        # Propogate through whole genes
        _ge, _ = self.T_whole(x = _gs)
        #whole_att_list.append(_att.detach().cpu().numpy()) 
        # Generate the genotype + response level embedding
        _fg, _ = self.T_reout(x = _ge)
        #whole_att_list.append(_att.detach().cpu().numpy()) 
        # Extract response class vector
        out = _fg[:, -1, :].contiguous().view(batch_len, -1)
        return out   

class NCIPREDICTOR(nn.Module):
    def __init__(self, num_of_genotypes = 1, num_of_dcls = 5, cond_dim = 128,  drug_dim = 128, device = 'cuda'):
        super(NCIPREDICTOR, self).__init__()    
        self.response_predictor = ResponsePredictor( cond_dim = cond_dim, drug_dim = drug_dim, device = device) 

        
        self.condition_encoder = RES_Condition_Encoder(num_of_genotypes=num_of_genotypes, num_of_dcls=num_of_dcls, device = device)
        
        print("Load pretrained cond_encoder ...") 
        pret_ckpt = torch.load("./data/model_ckpts/seed_44_0914_52.pth", map_location=device)
        self.condition_encoder.load_state_dict(pret_ckpt['condition_state_dict'], strict=False)
        for n, p in self.condition_encoder.named_parameters():
            if n == 'cls_embedding.weight':
                p.requires_grad = True
            else:
                p.requires_grad = False
            
        
        self.num_of_genotypes = num_of_genotypes
        self.num_of_dcls = num_of_dcls
        self.condim = cond_dim
        self.input_dim = drug_dim
        self.device_name = device
       
    def forward(self, batch):
        condition = self.condition_encoder(batch)
        auc = self.response_predictor((batch['drug'], condition))
        
        return auc
        