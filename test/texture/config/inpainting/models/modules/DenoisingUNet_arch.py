import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
import numpy as np
from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)
import matplotlib.pyplot as plt
import os
import os.path as osp

from torch.nn import init
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

                

import re
import torch.nn.functional as F
def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean)  / std#
    return output#,mean,std
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "position":
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 16

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
                
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class SPADEBlock(BaseNetwork):
    def __init__(self, fin, fout, semantic_nc, norm_G = 'spectralspadeposition3x3'):
        super(SPADEBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        hidden = fin
        # create conv layers
        self.conv_0 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, hidden, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, hidden, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, hidden, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)
            
        self.init_weights()
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        return  x+dx

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.upg = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out),
                SPADEBlock(dim_out,dim_out,1)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))
            

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)


    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

 
    
    def forward(self, xt, cond, time=-1, S = None):
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()
        t = self.time_mlp(time)

        h = [] 
        gh = []

        
        entropy = torch.zeros(xt.shape[0]).cuda()
            
        for b1, b2, attn, downsample, guide in self.downs:
            x = b1(x, t)
            h.append(x)
            gh.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)
            gh.append(x)

            x = downsample(x)
            
            x = guide(x, S)
 
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        
        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            
            x = attn(x)

            x = upsample(x)
        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return x


class ConditionalUNets(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out),
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time):#

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)

        h = []

        for b1, b2, attn, downsample  in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        return x


