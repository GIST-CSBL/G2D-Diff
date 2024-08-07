from vae_package import vocab, vae_util

import torch
from torch import nn, optim
from torch.nn import LayerNorm
import torch.nn.functional as F
import numpy as np
import math
import os

def vae_loss(x, x_out, mu, logvar, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    # x = x.long()[:,1:] - 1
    x = x.long()[:,1:]    # skip the start token
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD, BCE, KLD

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.loss_func = vae_loss

        ### Sequence length hard-coded into model
        # self.src_len = 127
        self.tgt_len = 126

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}

    def save(self, state, fn, path='ckpts', use_name=True):
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)
        pass

    def load(self, checkpoint_path, device):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
            
        self.loaded_from = checkpoint_path
        for k in self.current_state.keys():
            try:
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError:
                self.current_state[k] = None
        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
            else:
                pass
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.build_model()
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
        
    def train(self, train_mols, val_mols, epochs=100, save=True, save_freq=1, 
                log=True, log_dir='logs'):
        """
        Train model and validate(?)

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures (SMILES)
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions(?) of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        train_data = vae_util.vae_data_gen(train_mols, self.tgt_len, self.vo, self.smtk)
        # val_data = vae_data_gen(val_mols, SEQ_MAXLEN, vo, smtk)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=self.params['BATCH_SIZE'],
                                         shuffle=True, drop_last=True)
        kl_annealer = vae_util.KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                        epochs, self.params['ANNEAL_START'])
        log_freq = np.linspace(0, len(train_iter), 10)
        log_freq = log_freq.astype('int')

        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,kld_loss\n')
            log_file.close() 
            
        for epoch in range(epochs):
            self.model.train()
            beta = kl_annealer(epoch)
            
            losses = []
            bce_losses = []
            kld_losses = []
            for j, data in enumerate(train_iter):
                mols_data = data.to(self.device)
                src = mols_data.long()
                tgt = mols_data[:,:-1].long()

                x_out, mu, logvar = self.model(src, tgt)
                loss, bce, kld = self.loss_func(src, x_out, mu, logvar, beta)
                
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                
                kld = kld / beta
                avg_loss = np.mean(loss.item())
                avg_bce = np.mean(bce.item())
                avg_kld = np.mean(kld.item())
                
                losses.append(avg_loss)
                bce_losses.append(avg_bce)
                kld_losses.append(avg_kld)
                if log:
                    if j in log_freq:
                        log_file = open(log_fn, 'a')
                        log_file.write('{},{},{}, {:.5f},{:.5f},{:.5f}\n'.format(self.n_epochs, j, 'train', avg_loss, avg_bce, avg_kld))
                        log_file.close()
                    else:
                        pass
  
            epoch_loss = np.mean(losses)
            epoch_bce = np.mean(bce_losses)
            epoch_kld = np.mean(kld_losses)
            if log:
                log_file = open(log_fn, 'a')
                log_file.write(
                    '{},{},{},{:.5f},{:.5f},{:.5f}\n'.format(self.n_epochs, 'epoch_loss', 'train', epoch_loss,
                                                             epoch_bce, epoch_kld))
                log_file.close()

            self.n_epochs += 1
            print('Epoch - {} Loss - {:.5f} BCE - {:.5f} KLD - {:.5f} KLBeta - {:.6f}'.format(
                self.n_epochs, epoch_loss, epoch_bce, epoch_kld, beta))

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict
            
            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str)

    def calc_mems(self, data):
        """
        Method for calculating mem, mu, logvar

        Arguments:
            data (np.array, req): Input array containing SMILES strings
        Returns:
            repars(np.array): Reparameterized memory array
            mems(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        data = vae_util.vae_data_gen(data, self.tgt_len, self.vo, self.smtk)

        data_iter = torch.utils.data.DataLoader(data, batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, drop_last=False)

        mems, mus, logvars = [], [], []
        self.model.eval()
        for j, data in enumerate(data_iter):
            batch_data = data.long()
            mols_data = batch_data.to(self.device)

            ### Run through encoder to get memory
            _, mu, logvar, mem = self.model.encode(mols_data)
            mems.append(mem)
            mus.append(mu)
            logvars.append(logvar)
            
        return torch.vstack(mems), torch.vstack(mus), torch.vstack(logvars)

class RNNEncoder(nn.Module):
    """
    Simple recurrent encoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.device = device
        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.linear = nn.Linear(size, size)
        self.z_means = nn.Linear(size, d_latent)
        self.z_var = nn.Linear(size, d_latent)
        self.norm = LayerNorm(size)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x, h = self.gru(x, h)
        hidden = h[-1,:,:]
        mem = self.linear(hidden)
        mem = self.norm(mem)

        mu, logvar = self.z_means(mem), self.z_var(mem)
        repar = self.reparameterize(mu, logvar)
        return repar, mu, logvar, mem
        
    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()

class RNNDecoder(nn.Module):
    """
    Simple recurrent decoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, tgt_length, tf, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = tgt_length+1
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.device = device

        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.unbottleneck = nn.Linear(d_latent, size)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        h = self.initH(mem.shape[0])
        embedded = self.dropout(tgt)
        mem = F.relu(self.unbottleneck(mem))
        mem = mem.unsqueeze(1).repeat(1, self.max_length, 1)
        mem = self.norm(mem)
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)

class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        # self.proj = nn.Linear(d_model, vocab-1)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class RNNEncoderDecoder(nn.Module):
    """
    Recurrent Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder, src_embed, generator, params):
        super().__init__()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.generator = generator
        #### self.tgt_embed has been removed !!

    def forward(self, src, tgt):
        repar, mu, logvar, _ = self.encode(src)
        x, h = self.decode(tgt, repar)
        x = self.generator(x)
        return x, mu, logvar

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, repar):
        return self.decoder(self.src_embed(tgt), repar)

class AdamOpt:
    "Adam optimizer wrapper"
    def __init__(self, params, lr, optimizer):
        self.optimizer = optimizer(params, lr)
        self.state_dict = self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

class RNNVAE(VAEShell):
    """
    RNN-based VAE without attention.
    """
    def __init__(self, vo: vocab.Vocabulary, smtk: vocab.SmilesTokenizer, 
                 params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 device='cpu', load_fn=None):
        super().__init__(params, name)
        self.device = device    #### new way of assigning device

        #### new stuffs
        self.vo = vo
        self.smtk = smtk

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'rnn'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
             self.load(load_fn, self.device)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        #### device is assigned at __init__()
        encoder = RNNEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.device)
        decoder = RNNDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.tgt_len, self.params['teacher_force'],
                             self.device)
                             #### previously, tgt_length value was hard-coded to 125 here.
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, generator, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)
        self.model.to(self.device)

    def decode_from_z(self, z, method='greedy'):
        """
        Method for decoding given z vectors into SMILES strings

        Arguments:
            z (np.array): [N x d_latent] matrix
            method (str): greedy | multin

        Returns:
            tgt (torch.Tensor): Decoded smiles as token indices
            probs (torch.Tensor): [N x max_len x vocab_size] probablity from softmax output.
                The first position prob is always zeros (not used).
        """
        beg_idx = self.vo.get_BEG_idx()
        eos_idx = self.vo.get_EOS_idx()
        pad_idx = self.vo.get_PAD_idx()

        z = torch.tensor(z).to(self.device).float()
        N, _ = z.shape

        self.model.eval()
        tgt = torch.full((N, self.tgt_len+1), pad_idx).to(self.device)
        probs = torch.zeros((N, self.tgt_len+1, self.vocab_size))
        # put <BEG> at the start
        tgt[:,0] = beg_idx
        for i in range(self.tgt_len):
            out, _ = self.model.decode(tgt, z)  # x, h
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1)
            if method == 'greedy':
                _, next_word = torch.max(prob, dim=1)
            elif method == 'multin':
                next_word = torch.multinomial(prob, num_samples=1).view(-1)
            tgt[:,i+1] = next_word
            probs[:,i+1,:] = prob
        return tgt, probs

    def get_mem(self, src):
        self.model.eval()
        _, mu, log_var, mem = self.model.encode(src)
        return mu, mem
