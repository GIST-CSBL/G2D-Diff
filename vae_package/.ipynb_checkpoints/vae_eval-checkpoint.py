from vae_package import vae_model, vocab, vae_tool, pack_global, analysis_mp, vae_util
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

def evaluate_model_rand(rvae: vae_model.RNNVAE, sz: int, trainset_path: str,
                        fname: str, batch_size=32):
    """
        Arguments:
            sz: sample size used for evaluation
    """
    vo = rvae.vo
    sampler = vae_tool.RNNVAESampler(rvae, vo, batch_size)
    with open(trainset_path) as f:
        tr_smiles = [line.strip() for line in f.readlines()]

    #########
    # evaluate on random generations
    smiles_list, _ = sampler.sample_randn_smiles(sz, method='greedy')
    vacans, invids = analysis_mp.get_valid_canons(smiles_list)

    vld, unq, nvl = analysis_mp.evaluation_basic(sz, vacans, tr_smiles)
    lines = ["validity,uniqueness,novelty", '{:.5f},{:.5f},{:.5f}'.format(vld, unq, nvl)]
    with open(fname + "_rand_prf.txt", 'w') as f:
        f.writelines([line + '\n' for line in lines])

    rand_eval = generation_df(smiles_list, invids, vacans)
    rand_eval.to_csv(fname + "_rand.csv", index=False)


def evaluate_model_recon(rvae: vae_model.RNNVAE, sz: int, trainset_path: str,
                         fname: str, batch_size=32, repeat=1):
    vo = rvae.vo
    sampler = vae_tool.RNNVAESampler(rvae, vo, batch_size)
    with open(trainset_path) as f:
        tr_smiles = [line.strip() for line in f.readlines()]

    ##########
    # pick seed molecules from the training set, evaluate reconstructed smiles
    seeds = np.random.choice(tr_smiles, sz, replace=False)
    rep_seeds = np.repeat(seeds, repeats=repeat, axis=0)
    smiles_list, _ = sampler.sample_recon(seeds, varopt='mu', method='greedy', repeat=repeat)
    if repeat>1:
        smiles_list = list(reduce(operator.add, smiles_list))
    vacans, invids = analysis_mp.get_valid_canons(smiles_list)

    vld, unq, nvl = analysis_mp.evaluation_basic(sz*repeat, vacans, tr_smiles)
    lines = ["validity,uniqueness,novelty", '{:.5f},{:.5f},{:.5f}'.format(vld, unq, nvl)]
    with open(fname + "_recon_prf.txt", 'w') as f:
        f.writelines([line + '\n' for line in lines])

    recon_eval = generation_df(smiles_list, invids, vacans)
    recon_eval['seed'] = rep_seeds
    recon_eval.to_csv(fname + "_recon.csv", index=False)


def evaluate_model_mem(rvae: vae_model.RNNVAE, sz: int, trainset_path: str,
                       fname: str, batch_size=32, repeat=1):
    vo = rvae.vo
    sampler = vae_tool.RNNVAESampler(rvae, vo, batch_size)
    with open(trainset_path) as f:
        tr_smiles = [line.strip() for line in f.readlines()]

    ##########
    # pick seed molecules from the training set, evaluate reconstructed smiles
    seeds = np.random.choice(tr_smiles, sz, replace=False)
    rep_seeds = np.repeat(seeds, repeats=repeat, axis=0)

    mems_list = []
    train_data = vae_util.vae_data_gen(seeds, rvae.tgt_len, vo, rvae.smtk)
    data_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    rvae.model.eval()
    for i, data in enumerate(data_iter):
        mols_data = data.to(rvae.device)
        src = mols_data.long()

        mems = rvae.get_mem(src)
        mems_list.extend(mems.detach())

    smiles_list = sampler.sample_from_mem(mems_list, method='greedy', repeat=repeat)
    if repeat>1:
        smiles_list = list(reduce(operator.add, smiles_list))
    vacans, invids = analysis_mp.get_valid_canons(smiles_list)

    vld, unq, nvl = analysis_mp.evaluation_basic(sz*repeat, vacans, tr_smiles)
    lines = ["validity,uniqueness,novelty", '{:.5f},{:.5f},{:.5f}'.format(vld, unq, nvl)]
    with open(fname + "_mem_prf.txt", 'w') as f:
        f.writelines([line + '\n' for line in lines])

    recon_eval = generation_df(smiles_list, invids, vacans)
    recon_eval['seed'] = rep_seeds
    recon_eval.to_csv(fname + "_mem.csv", index=False)