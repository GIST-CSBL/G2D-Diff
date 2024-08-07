import torch
from torch import nn 
import numpy as np
from typing import List, Dict
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, dropout_rate = (0.2, 0.2)):
        super(MultiHeadAttention, self).__init__()   
        self.num_heads = num_heads
        self.d_head = d_head
        self.proj_q = nn.Linear(d_model, num_heads * d_head)
        self.proj_k = nn.Linear(d_model, num_heads * d_head)
        self.proj_v = nn.Linear(d_model, num_heads * d_head)
        self.proj_o = nn.Linear(num_heads * d_head, d_model)
        self.dropout1 = nn.Dropout(dropout_rate[0])
        self.dropout2 = nn.Dropout(dropout_rate[1])
        
    def forward(self, q, k, v, mask = None):
        device = q.device
        batch_size = q.shape[0]
        
        queries = self.proj_q(q).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        keys = self.proj_k(k).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        values = self.proj_v(v).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(queries.size(-1)))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
            
        alpha = self.dropout1(torch.softmax(scores, dim=-1))
        out = torch.matmul(alpha, values).transpose(1, 2)
        out = self.dropout2(self.proj_o(out.contiguous().view(batch_size, -1, self.num_heads * self.d_head)))
        
        return out, alpha
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model_in, d_model_out, dropout_rate = (0.2, 0.2)):
        super(PositionWiseFeedForward, self).__init__()
        self.W1 = nn.Linear(d_model_in, d_model_out)
        self.activation = nn.GELU()
        self.W2 = nn.Linear(d_model_out, d_model_in)
        self.dropout1 = nn.Dropout(dropout_rate[0])
        self.dropout2 = nn.Dropout(dropout_rate[1])
        
    def forward(self, x):
        return self.dropout2(self.W2(self.dropout1(self.activation(self.W1(x)))))
    
class TFblock(nn.Module):
    def __init__(self, d_model, num_heads, d_head, mha_drate = (0.2, 0.2), pff_drate = (0.2, 0.2)):
        super(TFblock, self).__init__()
        self.mha = MultiHeadAttention(d_model = d_model, num_heads = num_heads, d_head = d_head, dropout_rate = mha_drate)
        self.pFF = PositionWiseFeedForward(d_model_in = d_model, d_model_out = d_model * 4, dropout_rate = pff_drate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
               
    def forward(self, x, mask = None):
        emb, att = self.mha(x, x, x, mask)
        x = self.norm1(x + emb)
        x = self.norm2(x + self.pFF(x))
        return x, att           

class Mut2Signal(nn.Module):
    def __init__(self, emb_size = 128):
        super(Mut2Signal, self).__init__()
        self.gene2weight = nn.Linear(emb_size, emb_size)
        self.gene2bias = nn.Linear(emb_size, emb_size)
        
    def forward(self, gene_emb, mut_emb):
        gene_weight = self.gene2weight(gene_emb)
        gene_bias = self.gene2bias(gene_emb)
        return (gene_weight * mut_emb) + gene_bias
    
class Condition_Encoder(nn.Module):
    def __init__(self, num_of_genotypes, num_of_dcls, num_of_genes = 718,  gene_emb_size = 128,  device = 'cuda', neighbor_info = True):
        super(Condition_Encoder, self).__init__()
        
        self.num_of_genes = num_of_genes
        self.gene_emb_size = gene_emb_size
        self.device = device
        
        self.gene_embedding = nn.Embedding(num_of_genes, gene_emb_size)
        self.dcls_embedding = nn.Embedding(num_of_dcls, gene_emb_size)
        self.muts_embedding = nn.Embedding(num_of_genotypes, gene_emb_size)
        
        self.mut2signal = Mut2Signal(emb_size = gene_emb_size)
        self.T_neigh = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
        self.T_whole = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
        self.T_reout = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
        
        ############################################################################################################################################
        
        if neighbor_info:
            print("NeST neighbor info is used")
            with open("NeST_neighbor_adj.npy", "rb") as f:
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
        _dclas = self.dcls_embedding(data['class']).view(batch_len, 1, -1)
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
        gsj_cat = torch.cat([_gene_add, _dclas], axis = 1)
        # Propogate through NeST siblings
        _gs, _att = self.T_neigh(x = gsj_cat, mask = revised_adj)
        whole_att_list.append(_att.detach().cpu()) 
        # Propogate through whole genes
        _ge, _att = self.T_whole(x = _gs)
        whole_att_list.append(_att.detach().cpu()) 
        # Generate the genotype + response level embedding
        _fg, _att = self.T_reout(x = _ge)
        whole_att_list.append(_att.detach().cpu()) 
        # Extract response class vector
        out = _fg[:, -1, :].contiguous().view(batch_len, -1)
        attention = _att[:, :, -1, :].squeeze().detach().cpu()
        return _ge, out, attention, whole_att_list
    
    
class DrugEncoder(nn.Module):
    def __init__(self, input_dim = 128, device = 'cuda'):
        super(DrugEncoder, self).__init__()
        
        self.sub_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    
    def forward(self, input_vec):
        x = self.sub_layer(input_vec)
        return x    


# import torch
# from torch import nn 
# import numpy as np
# from typing import List, Dict
# import torch.nn.functional as F

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, d_head, dropout_rate = 0.):
#         super(MultiHeadAttention, self).__init__()   
#         self.num_heads = num_heads
#         self.d_head = d_head
#         self.proj_q = nn.Linear(d_model, num_heads * d_head)
#         self.proj_k = nn.Linear(d_model, num_heads * d_head)
#         self.proj_v = nn.Linear(d_model, num_heads * d_head)
#         self.proj_o = nn.Linear(num_heads * d_head, d_model)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#     def forward(self, q, k, v, mask = None):
#         device = q.device
#         batch_size = q.shape[0]
        
#         queries = self.proj_q(q).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
#         keys = self.proj_k(k).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
#         values = self.proj_v(v).contiguous().view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
#         scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(queries.size(-1)))
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -float('inf'))
            
#         alpha = self.dropout1(torch.softmax(scores, dim=-1))
#         out = torch.matmul(alpha, values).transpose(1, 2)
#         out = self.dropout2(self.proj_o(out.contiguous().view(batch_size, -1, self.num_heads * self.d_head)))
        
#         return out, alpha
    
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model_in, d_model_out, dropout_rate = 0.):
#         super(PositionWiseFeedForward, self).__init__()
#         self.W1 = nn.Linear(d_model_in, d_model_out)
#         self.activation = nn.GELU()
#         self.W2 = nn.Linear(d_model_out, d_model_in)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#     def forward(self, x):
#         return self.dropout2(self.W2(self.dropout1(self.activation(self.W1(x)))))
    
# class TFblock(nn.Module):
#     def __init__(self, d_model, num_heads, d_head, mha_drate = 0.2, pff_drate = 0.2):
#         super(TFblock, self).__init__()
#         self.mha = MultiHeadAttention(d_model = d_model, num_heads = num_heads, d_head = d_head, dropout_rate = mha_drate)
#         self.pFF = PositionWiseFeedForward(d_model_in = d_model, d_model_out = d_model * 4, dropout_rate = pff_drate)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
               
#     def forward(self, x, mask = None):
#         emb, att = self.mha(x, x, x, mask)
#         x = self.norm1(x + emb)
#         x = self.norm2(x + self.pFF(x))
#         return x, att           

# class Mut2Signal(nn.Module):
#     def __init__(self, emb_size = 128, dropout_rate = 0.2):
#         super(Mut2Signal, self).__init__()
#         self.gene2weight = nn.Linear(emb_size, emb_size)
#         self.gene2bias = nn.Linear(emb_size, emb_size)
#         self.dropout = nn.Dropout(dropout_rate)
#     def forward(self, gene_emb, mut_emb):
#         gene_weight = self.gene2weight(gene_emb)
#         gene_bias = self.gene2bias(gene_emb)
#         return self.dropout((gene_weight * mut_emb) + gene_bias)
    
# class Condition_Encoder(nn.Module):
#     def __init__(self, num_of_genotypes, num_of_dcls, num_of_genes = 718,  gene_emb_size = 128,  device = 'cuda', neighbor_info = True):
#         super(Condition_Encoder, self).__init__()
        
#         self.num_of_genes = num_of_genes
#         self.gene_emb_size = gene_emb_size
#         self.device = device
        
#         self.gene_embedding = nn.Embedding(num_of_genes, gene_emb_size)
#         self.dcls_embedding = nn.Embedding(num_of_dcls, gene_emb_size)
#         self.muts_embedding = nn.Embedding(num_of_genotypes, gene_emb_size)
        
#         self.mut2signal = Mut2Signal(emb_size = gene_emb_size)
#         self.T_neigh = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
#         self.T_whole = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8)) 
#         self.T_reout = TFblock(d_model = gene_emb_size, num_heads = 8, d_head = (gene_emb_size // 8))
        
#         ############################################################################################################################################
        
#         if neighbor_info:
#             print("NeST neighbor info is used")
#             with open("NeST_neighbor_adj.npy", "rb") as f:
#                 self.gene_adj = torch.BoolTensor(np.load(f)).to(device)
#         else: 
#             print("No prior knowledge is used")
#             self.gene_adj = None

#     def get_new_adj(self, adj):
#         new_adj = torch.ones(self.num_of_genes + 1, self.num_of_genes + 1).to(self.device)
#         new_adj[:-1, :-1] = adj
#         return new_adj


#     def forward(self, data: Dict):

#         batch_len = len(data['class'])
#         _genotype = data['genotype']
        
#         gene_list = np.array([x for x in range(self.num_of_genes)])
#         _genes = self.gene_embedding(torch.IntTensor(gene_list).to(self.device)).unsqueeze(0).expand(batch_len,-1, -1)
#         _dclas = self.dcls_embedding(data['class']).view(batch_len, 1, -1)
#         revised_adj = self.get_new_adj(self.gene_adj)

#         _gene_add = _genes
#         whole_att_list = []
#         mut_keys = list(_genotype.keys()) # ['MUT', 'CNA', 'CND']
#         for i, mut_type in enumerate(mut_keys):
#             # Generate the mutation embedding
#             # Normalize + transform to each gene's perspective
#             mut_list = np.array([i for _ in range(self.num_of_genes)])
#             _mut_base = self.muts_embedding(torch.IntTensor(mut_list).to(self.device)).unsqueeze(0).expand(batch_len,-1, -1)
#             mut_emb_affn = self.mut2signal(_genes, _mut_base)
#             # Generate mutation mask
#             mut_emb_mask = _genotype[mut_type].unsqueeze(-1).expand(-1,-1, self.gene_emb_size)
#             # Add the mutation signal to the gene embedding
#             _gene_add = _gene_add + (mut_emb_mask * mut_emb_affn)
            
#         ########################################## PROPOGATION BEGIN ##########################################
#         # Concat response class vector
#         gsj_cat = torch.cat([_gene_add, _dclas], axis = 1)
#         # Propogate through NeST siblings
#         _gs, _ = self.T_neigh(x = gsj_cat, mask = revised_adj)
#         #whole_att_list.append(_att.detach().cpu()) 
#         # Propogate through whole genes
#         _ge, _ = self.T_whole(x = _gs)
#         #whole_att_list.append(_att.detach().cpu()) 
#         # Generate the genotype + response level embedding
#         _fg, _ = self.T_reout(x = _ge)
#         #whole_att_list.append(_att.detach().cpu()) 
#         # Extract response class vector
#         out = _fg[:, -1, :].contiguous().view(batch_len, -1)
#         #attention = _att[:, :, -1, :].squeeze().detach().cpu()
#         return _ge, out, _, _
    
    
# class DrugEncoder(nn.Module):
#     def __init__(self, input_dim = 128, device = 'cuda'):
#         super(DrugEncoder, self).__init__()
        
#         self.sub_layer = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.LayerNorm(input_dim),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(input_dim, input_dim),
#             nn.LayerNorm(input_dim),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(input_dim, input_dim),
#             nn.LayerNorm(input_dim)
#         )

    
#     def forward(self, input_vec):
#         x = self.sub_layer(input_vec)
#         return x    
    




