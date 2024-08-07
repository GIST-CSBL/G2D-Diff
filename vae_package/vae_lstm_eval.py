from vae_package import pack_global, analysis_mp, vae_util
from vae_package import vae_lstm_tool as vae_tool
# from vae_package import vae_model
from vae_package import vae_lstm_model as vae_model
import pandas as pd
import numpy as np
import torch
from functools import reduce
import operator

def generation_df(smiles_list, invids, vacans):
    """
        Arguments:
            smiles_list: generated smiles
            invids (np.array): position of invalid smiles generations
            vacans: list of valid & canonicalized smiles
    """
    sz = len(smiles_list)
    gendf = pd.DataFrame(columns=['val_bin', 'sample', 'canon'])
    gendf['sample'] = smiles_list
    val_bins = np.ones(sz).astype(int)
    val_bins[invids] = 0
    gendf['val_bin'] = val_bins
    canon = np.array(['x' for i in range(sz)], dtype=object)
    canon[val_bins.astype(bool)] = vacans
    gendf['canon'] = canon
    return gendf

def evaluate_generation(gens, trainset, fpath:str):
    vacans, invids = analysis_mp.get_valid_canons(gens)
    vld, unq, nvl = analysis_mp.evaluation_basic(len(gens), vacans, trainset)
    lines = ["validity,uniqueness,novelty", '{:.5f},{:.5f},{:.5f}'.format(vld, unq, nvl)]
    with open("result/" + fpath, 'w') as f:
        f.writelines([line + '\n' for line in lines])
    return vacans, invids
    
def evaluate_model_rand(rvae: vae_model.RNNVAE, sz: int, data_smiles, fname, batch_size=32):
    """
        Arguments:
            sz: sample size used for evaluation
    """
    sampler = vae_tool.RNNVAESampler(rvae, rvae.vo, batch_size)
   
    #########
    # evaluate on random generations
    smiles_list, _ = sampler.sample_randn_smiles(sz, method='greedy')
    vacans, invids = evaluate_generation(smiles_list, data_smiles, fpath=fname+"_rand_perf.txt")
    rand_eval = generation_df(smiles_list, invids, vacans)
    rand_eval.to_csv('result/' + fname + "_rand.csv", index=False)
    return

def evaluate_model_recon(rvae: vae_model.RNNVAE, sz: int, data_smiles, fname, batch_size=32, repeat=1):
    """
        Arguments:
            data_smiles: list of smiles from where the seeds are selected,
                and also used to calculate novelty.
    """
    sampler = vae_tool.RNNVAESampler(rvae, rvae.vo, batch_size)
    
    ##########
    # pick seed molecules from the training set, evaluate reconstructed smiles
    seeds = np.random.choice(data_smiles, sz, replace=False)
    smiles_list, _ = sampler.sample_recon(seeds, varopt='mu', method='greedy', repeat=repeat)
    if repeat>1:
        smiles_list = list(reduce(operator.add, smiles_list))

    vacans, invids = evaluate_generation(smiles_list, data_smiles, fpath=fname+"_recon_perf.txt")
    recon_eval = generation_df(smiles_list, invids, vacans)

    rep_seeds = np.repeat(seeds, repeats=repeat, axis=0)
    recon_eval['seed'] = rep_seeds
    recon_eval.to_csv('result/' + fname + "_recon.csv", index=False)
    return

def evaluate_model_mem(rvae: vae_model.RNNVAE, sz: int, data_smiles, fname, batch_size=32, repeat=1):
    sampler = vae_tool.RNNVAESampler(rvae, rvae.vo, batch_size)

    ##########
    # pick seed molecules from the training set, evaluate reconstructed smiles
    seeds = np.random.choice(data_smiles, sz, replace=False)
    mems_list = []
    train_data = vae_util.vae_data_gen(seeds, rvae.tgt_len, rvae.vo, rvae.smtk)
    data_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    rvae.model.eval()
    for i, data in enumerate(data_iter):
        mols_data = data.to(rvae.device)
        src = mols_data.long()
        mems = rvae.get_mem(src)
        mems_list.extend(mems.detach().cpu().numpy())
    smiles_list = sampler.sample_from_mem(np.array(mems_list), method='greedy', repeat=repeat)
    if repeat>1:
        smiles_list = list(reduce(operator.add, smiles_list))

    vacans, invids = evaluate_generation(smiles_list, data_smiles, fpath=fname+"_mem_perf.txt")
    recon_eval = generation_df(smiles_list, invids, vacans)
    
    rep_seeds = np.repeat(seeds, repeats=repeat, axis=0)
    recon_eval['seed'] = rep_seeds
    recon_eval.to_csv('result/' + fname + "_mem.csv", index=False)
    return

def validation_perf(rvae: vae_model.RNNVAE, valdationset_path: str, batch_size=32):
    rvae.model.eval()

    with open(valdationset_path, 'r') as f:
        vl_smiles = [line.strip() for line in f.readlines()]
    vl_data = vae_util.vae_data_gen(vl_smiles, rvae.tgt_len, rvae.vo, rvae.smtk)
    data_iter = torch.utils.data.DataLoader(vl_data, batch_size=batch_size, shuffle=False, drop_last=True)

    losses = []
    bce_losses = []
    kld_losses = []
    for j, data in enumerate(data_iter):
        mols_data = data.to(rvae.device)
        src = mols_data.long()
        tgt = mols_data[:,:-1].long()

        x_out, mu, logvar = rvae.model(src, tgt)
        loss, bce, kld = rvae.loss_func(src, x_out, mu, logvar, beta=1.0)

        avg_loss = np.mean(loss.item())
        avg_bce = np.mean(bce.item())
        avg_kld = np.mean(kld.item())

        losses.append(avg_loss)
        bce_losses.append(avg_bce)
        kld_losses.append(avg_kld)

    return np.mean(losses), np.mean(bce_losses), np.mean(kld_losses)

def evaluate_model(rvae: vae_model.RNNVAE, sz: int, trainset_path: str,
                    vldset_path: str, fname: str, batch_size=32):
    """
        This method evaluate the model's generation on random, reconstruction, and memory-input cases.
        Arguments:
            sz: sample size used for evaluation
    """
    with open(trainset_path, 'r') as f:
        tr_smiles = [line.strip() for line in f.readlines()]

    #########
    # evaluate validation performance
    print("validation ...")
    loss, bce, kld = validation_perf(rvae, vldset_path, batch_size=batch_size)
    lines = ["loss,bce,kld","{:.5f},{:.5f},{:.5f}".format(loss,bce,kld)]
    with open('result/'+fname+'_vldprf.txt', 'w') as f:
        f.writelines([line+'\n' for line in lines])

    #########
    # evaluate on random generations
    print("random generation ...")
    evaluate_model_rand(rvae, sz, data_smiles=tr_smiles, fname=fname, batch_size=batch_size)

    ##########
    # pick seed molecules from the training set, evaluate reconstructed smiles
    print("reconstruction ...")
    evaluate_model_recon(rvae, sz, data_smiles=tr_smiles, fname=fname, batch_size=batch_size)

    ##########
    # evaluate memory reconstruction
    print("memory reconstruction ...")
    evaluate_model_mem(rvae, sz, data_smiles=tr_smiles, fname=fname, batch_size=batch_size)
    return