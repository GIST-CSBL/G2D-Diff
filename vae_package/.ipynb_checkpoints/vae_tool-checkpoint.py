from vae_package import vae_model, vocab, vae_util
import torch
import numpy as np

class RNNVAESampler:
    """

    """
    def __init__(self, rvae: vae_model.RNNVAE, vo: vocab.Vocabulary, batch_size: int):
        self.rvae = rvae
        self.vo = vo
        self.bs = batch_size
    
    def sample_from_mem(self, mems: np.array, method='greedy', repeat=1):
        """ 
            Sample SMILES from mem vectors
            Arguments:
                mems : [N x d] N mem vectors
                repeat : how many reparameterization repeat for a single mem vector
            Returns:
                smiles_list: (if repeat > 1) [N x repeat x seq_len]
                        (if repeat == 1) [N x seq_len]
        """
        self.rvae.model.eval()
        encoder = self.rvae.model.encoder

        N = len(mems)
        rep_mems = np.repeat(mems, repeats=repeat, axis=0)

        smiles_list = []
        data_iter = torch.utils.data.DataLoader(rep_mems, batch_size=self.bs, shuffle=False)
        for i, data in enumerate(data_iter):
            _mem = data.to(self.rvae.device)
            _mu, _logvar = encoder.z_means(_mem), encoder.z_var(_mem)
            _repar = encoder.reparameterize(_mu, _logvar)    # equivalent to the z vectors
            gen = self.sample_from_z(_repar, method)

            gen = gen.numpy()
            smiles_list.extend(self.gen_to_smiles(gen))

        if repeat > 1:
            smiles_list = np.array(smiles_list).reshape((N,repeat))
            smiles_list = smiles_list.tolist()
        return smiles_list

    def sample_from_z(self, z: np.array, method='greedy'):
        """ The output format is the integer(token index) matrix. torch.Tensor[N x maxlen] """
        pad_idx = self.vo.get_PAD_idx()
        N, _ = z.shape
        
        # fill with <PAD>
        generation = torch.full((N, self.rvae.tgt_len), pad_idx, dtype=torch.long)
        sampled_count = 0
        while sampled_count < N:
            until = sampled_count + self.bs
            z_b = z[sampled_count:until]
            decoded, _ = self.rvae.decode_from_z(z_b, method=method)
            decoded = decoded[:,1:].detach().cpu()
            generation[sampled_count:until] = decoded
            sampled_count = until
        return generation

    def sample_randn(self, num_samples: int, method='greedy'):
        """ The output gen is the integer(token index) matrix. torch.Tensor[N x maxlen] """
        z = np.random.randn(num_samples, self.rvae.params['d_latent'])
        gen = self.sample_from_z(z, method=method)
        return gen, z

    def gen_to_smiles(self, gen: np.array):
        """ 
            Arguments:
                gen (np.array): [N x L] N examples of smiles indices
            Returns:
                smiles_list (list): list of SMILES strings (truncate from the first special token encounter)
        """
        truncated = vae_util.truncate_specials(gen, self.vo)
        smiles_list = []
        for ex in truncated:
            smiles_list.append(self.vo.decode(ex))
        return smiles_list

    def sample_randn_smiles(self, num_samples: int, method='greedy'):
        """ 
            Sample SMILES from random normal codes.
            Returns:
                smiles_list (list): list of SMILES strings (truncate from the first special token encounter)
                z (np.array): z vectors that are used to be decoded.
        """
        gen, z = self.sample_randn(num_samples, method=method)
        gen = gen.numpy()
        smiles_list = self.gen_to_smiles(gen)
        return smiles_list, z

    def sample_recon(self, input_smiles, varopt, method='greedy', repeat = 1):
        data = vae_util.vae_data_gen(input_smiles, self.rvae.tgt_len, self.vo, self.rvae.smtk)
        data_iter = torch.utils.data.DataLoader(data, batch_size=self.bs,
                                                shuffle=False, drop_last=False)
        smiles_repeat_list = []
        z_repeat_list = []
        self.rvae.model.eval()

        for i in range(repeat):
            smiles_list = []
            z_list = []
            for j, data in enumerate(data_iter):
                mols_data = data.long().to(self.rvae.device)

                _, mu, logvar, _ = self.rvae.model.encode(mols_data)

                if varopt == 'mu':
                    z = mu
                else:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std

                z = z.detach()
                z_list.append(z)
                gen = self.sample_from_z(z, method=method)
                smiles = self.gen_to_smiles(gen.cpu().numpy())
                smiles_list += smiles
            smiles_repeat_list.append(smiles_list)
            z_repeat_list.append(torch.vstack(z_list))

        if repeat == 1:
            reconstructed_smiles = smiles_repeat_list[0]
            reconstructed_z = z_repeat_list[0].cpu()
        else:
            reconstructed_smiles = np.vstack(smiles_repeat_list).T.tolist()
            reconstructed_z = torch.vstack(z_repeat_list).reshape(repeat, len(input_smiles), -1).cpu()

        return reconstructed_smiles, reconstructed_z

