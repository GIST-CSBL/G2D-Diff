from vae_package import vocab
import torch
import numpy as np

class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        else:
            pass
        return beta

class CyclicAnnealer:

    def __init__(self, kl_low, kl_high, period, method='linear', low_len=0, high_len=0):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.period = period
        self.anneal_period = period - low_len - high_len
        kl_lows = [self.kl_low] * low_len
        kl_highs = [self.kl_high] * high_len

        if method == 'sigmoid':
            center = int(self.anneal_period / 2)
            kl_steps = [((self.kl_high - self.kl_low) / (1.0 + np.exp(center - i)) + self.kl_low) for i in
                        range(self.anneal_period)]

        else:
            step = (self.kl_high - self.kl_low) / (self.anneal_period - 1)
            kl_steps = [self.kl_low + step * i for i in range(self.anneal_period)]

        kl_list = kl_lows + kl_steps + kl_highs
        self.kl_list = kl_list

    def __call__(self, epoch):
        idx = epoch % self.period
        return self.kl_list[idx]


def vae_data_gen(smiles, seq_maxlen, vo: vocab.Vocabulary, smtk: vocab.SmilesTokenizer):
    """
    Encodes input smiles to tensors with token ids.
    <BEG> token is added at the start, <EOS> token is added at the end.

    Arguments:
        smiles (list, req): list containing input molecular structures
        seq_maxlen (int, req): max SMILES sequence size, before adding <BEG> and <EOS>
        vo (vocab.Vocabulary) : ...
        smtk (vocab.SmilesTokenizer) : ...
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    """
    beg_idx = vo.get_BEG_idx()
    eos_idx = vo.get_EOS_idx()
    pad_idx = vo.get_PAD_idx()
    
    enc_seqlen = seq_maxlen+2
    encoded_data = torch.full((len(smiles), enc_seqlen), pad_idx) # fill the matrix with <PAD>
    for j, smi in enumerate(smiles):
        enc_smi = vo.encode(smtk.tokenize(smi))[:seq_maxlen]
        enc_smi = np.concatenate([[beg_idx],enc_smi,[eos_idx]])
        encoded_data[j,:len(enc_smi)] = torch.tensor(enc_smi)
        
    return encoded_data

def truncate_specials(decoded, vo: vocab.Vocabulary):
    """
        Truncate smiles token indices from the tailing (first-encountered) special token.
        Arguments:
            decoded (np.array): [N x L] N examples of smiles indices
            vo (vocab.Vocabulary) : ...
        Returns:
            truncated (list): list of numpy array
    """
    beg_idx = vo.get_BEG_idx()
    eos_idx = vo.get_EOS_idx()
    pad_idx = vo.get_PAD_idx()
    N, _ = decoded.shape
    truncated = []
    for i in range(N):
        tinds = decoded[i]
        poss = vo.locate_tokens(tinds, [beg_idx, eos_idx, pad_idx])
        if len(poss) > 0:
            truncated.append(tinds[:poss[0]])
        else:
            truncated.append(tinds[:])
    return truncated
