
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn 
from einops import rearrange, repeat, reduce
from functools import partial
import math
import random
import os
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from src.g2d_diff_ce import *



class ConditioningBlock(nn.Module):
    def __init__(self, emb_dim = 128):
        super(ConditioningBlock, self).__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.InstanceNorm1d(emb_dim)
        self.act = nn.GELU()
        
        self.scale_shift = nn.Sequential(
                                nn.GELU(),
                                nn.Linear(emb_dim * 2, emb_dim * 2)
        )
        
        self.mlp = nn.Sequential(nn.GELU(),
                                 nn.Linear(emb_dim, emb_dim))
        
    def forward(self, xemb, cemb, temb):
        scale_shift = self.scale_shift(torch.cat([temb, cemb], dim = -1))
        scale, shift = scale_shift[:, :self.emb_dim], scale_shift[:, self.emb_dim:]
        
        x_proj_norm = self.norm(self.proj(xemb))
        x_ss = self.mlp(scale * x_proj_norm + shift)
        
        
        return self.act(x_ss)
    
    
        
class EPSModel(nn.Module):
    def __init__(self, emb_dim = 128, device = 'cuda', training = True, w = 0, prand = 0.05, layers = 6):
        super(EPSModel, self).__init__()
        self.device = device
        self.training = training
        self.w = w
        self.prand = prand
        
        self.layers = layers
        
        self.xemb_layer = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                       nn.GELU(),
                                       nn.Linear(emb_dim, emb_dim))
        
        self.cond_layer = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                       nn.GELU(),
                                       nn.Linear(emb_dim, emb_dim))
        
        self.time_layer = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                       nn.GELU(),
                                       nn.Linear(emb_dim, emb_dim))
        
                                        
                                        
        self.generator = nn.ModuleList([ConditioningBlock(emb_dim = emb_dim) for _ in range(self.layers)])
        
        self.final_layer = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                       nn.GELU(),
                                       nn.Linear(emb_dim, emb_dim))
        
        
        self.initialize_parameters()
        self.emb_dim = emb_dim

        
        self.condition_encoder = Condition_Encoder(num_of_genotypes=3, num_of_dcls=5, device = device, get_att = False)
        
        print("Load pretrained cond_encoder ...") 
        pret_ckpt = torch.load("./data/model_ckpts/seed_44_0914_52.pth", map_location=device)
        self.condition_encoder.load_state_dict(pret_ckpt['condition_state_dict'])
        for n, p in self.condition_encoder.named_parameters():
            p.requires_grad = False
        self.condition_encoder.eval()
        
        
        

    def initialize_parameters(self):
        for layer in self.named_parameters():  
            if 'weight' in layer[0]:
                torch.nn.init.xavier_normal_(layer[1])

            elif 'bias' in layer[0]:
                torch.nn.init.zeros_(layer[1])    
        
    
    def get_timestep_embedding(self, timesteps):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=self.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb
    
    def forward(self, x, t, batch):
        pos_emb = self.get_timestep_embedding(t)
        temb = self.time_layer(pos_emb)
        xemb = self.xemb_layer(x)
                               
        _, cond, _, _ = self.condition_encoder(batch)
 
        if self.training:
            batch_size = batch['class'].shape[0]
            p_vec = (torch.rand(batch_size) > self.prand).to(torch.float32).to(self.device).unsqueeze(0).expand(self.emb_dim, -1).T
            rand_cond = cond * p_vec
            cemb = self.cond_layer(rand_cond)
            
     
            for i in range(self.layers):
                xemb = self.generator[i](xemb, cemb, temb)
                
            e_t = self.final_layer(xemb)
            
        if not self.training:
            xcemb = xemb
            cemb = self.cond_layer(cond)
            for i in range(self.layers):
                xcemb = self.generator[i](xcemb, cemb, temb)
            e_c = self.final_layer(xcemb)
            
            xnemb = xemb
            null_cond = torch.zeros_like(cond)
            ncmb = self.cond_layer(null_cond)
            for i in range(self.layers):
                xnemb = self.generator[i](xnemb, ncmb, temb)
            e_nc = self.final_layer(xnemb)
            
            e_t = e_c + self.w * (e_c-e_nc)
        
        return e_t

'''
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.long()
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(self, 
        n_steps = 300,
        training = True,
        cfgw = 0,
        prand = 0.1,
        device = 'cuda'

    ):
        super(Diffusion, self).__init__()
        self.model = EPSModel(training = training, w = cfgw, prand = prand, device = device)

        betas = cosine_beta_schedule(n_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        
        snr = alphas_cumprod / (1 - alphas_cumprod)

        self.register_buffer('loss_weight', snr.clone())
        
        self.num_timesteps = n_steps
 
 

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, batch):
        model_output = self.model(x, t, batch)
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
    
    def p_mean_variance(self, x, t, batch):
        _, pred_x_start = self.model_predictions(x, t, batch)
        x_start = pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t: int, batch):
        device = self.betas.device

        batch_size = batch['class'].shape[0]
        
        batched_times = torch.full((batch_size,), t, device = device, dtype = torch.int)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, batch = batch)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_vec = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_vec, x_start

    @torch.no_grad()
    def p_sample_loop(self, batch, rand_vec):
        device = self.betas.device

        batch_size = batch['class'].shape[0]

        vec = rand_vec

        x_start = None
        vec_list = []
        vec_list.append(vec)
        for t in reversed(range(0, self.num_timesteps)):
            vec, x_start = self.p_sample(vec, t, batch)
            vec_list.append(vec)

        return vec, vec_list
    @torch.no_grad()
    def ddim_sample(self, batch, sampling_eta = 0, sampling_time = 100):
        batch_size, device, total_timesteps, sampling_timesteps, eta = batch['class'].shape[0], self.betas.device, self.num_timesteps, sampling_time, sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        gen_drug = torch.randn((batch_size, 128), device = device)


        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.int)
 
            pred_noise, pred_x_start = self.model_predictions(gen_drug, time_cond, batch)

            if time_next < 0:
                gen_drug = pred_x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(gen_drug)
   
            gen_drug = pred_x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

    
        return gen_drug
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        return F.mse_loss
   

    def p_losses(self, batch, t, noise = None):
        x_start = batch['drug']
        batch_size, feat_dim = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly


        # predict and take gradient step

        model_out = self.model(x, t, batch)
        target = x_start
       

        loss = self.loss_fn(model_out, target, reduction = 'none')
        
        loss = reduce(loss, 'b ... -> b', 'mean')
        # SNR weighting
        lw = extract(self.loss_weight, t, loss.shape)
        loss = lw * loss
        return loss.mean() 

    def forward(self, batch):
        batch_size, feat_dim, device = batch['drug'].shape[0], batch['drug'].shape[1], batch['drug'].device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).float()

        return self.p_losses(batch, t)    


    
    
