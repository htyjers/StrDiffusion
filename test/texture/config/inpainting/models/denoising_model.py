import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def gray_to_edge(image, sigma):

    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

    return edge




class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        #if opt["dist"]:
            #self.rank = torch.distributed.get_rank()
        #else:
            #self.rank = -1  # non dist training
        train_opt = opt["train"]
        

        # define network and load pretrained models
        self.model, self.models, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.models = self.models.to(self.device)
        self.dis = self.dis.to(self.device)
        self.model = DataParallel(self.model)
        self.models = DataParallel(self.models)
        self.dis = DataParallel(self.dis)
        self.load()
        if self.is_train:
            self.model.train()
            self.models.train()
            self.dis.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss1 = nn.L1Loss(reduction='mean') 
            self.loss2 = nn.MSELoss()
            
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (k,v,) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))


            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()


    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        #if GT is not None: 
        self.state_0 = GT.to(self.device)  # GT
        self.mask = mask.to(self.device) # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)

    def optimize_parameters_sigle(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        
        self.optimizer.zero_grad()
        
        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps) ##产生想要structure.
        yt_1_expection  = torch.zeros_like(yt_1_optimum)
        timesteps = timesteps.to(self.device)
        # Get noise and score
        S_timestep, S_states = self.S_sde.generate_random_states_texture(x0=self.S_GT, mu=self.S_LQ * self.mask, timesteps = timesteps - 1)
        S_optimum = self.S_sde.reverse_optimum_step(S_states, self.S_GT, S_timestep)
        noise = sde.noise_fn(self.state, timesteps.squeeze(),S_optimum)
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)

        loss = self.weight * (self.loss_fn(yt_1_expection, yt_1_optimum, self.mask))
        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()


    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        
        self.optimizer.zero_grad()
        
        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps) ##产生想要structure.
        yt_1_expection  = torch.zeros_like(yt_1_optimum)
        timesteps = timesteps.to(self.device)
        # Get noise and score
        S_timestep, S_states = self.S_sde.generate_random_states_texture(x0=self.S_GT, mu=self.S_LQ * self.mask, timesteps = timesteps - 1)
        S_optimum = self.S_sde.reverse_optimum_step(S_states, self.S_GT, S_timestep)
        noise,g_score = sde.noise_fn(self.state, timesteps.squeeze(),S_optimum)
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        
        loss = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask)
        loss += 0.1 * (self.loss1(torch.ones_like(g_score*(1-self.mask)),g_score*(1-self.mask))+self.loss2(torch.ones_like(g_score)*(1-self.mask),g_score*(1-self.mask)))
        loss += self.loss1(yt_1_expection * g_score + (1 - g_score) * yt_1_optimum, yt_1_optimum)
        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def begin(self,sde,noisy_states,S_sde,X_GT,mask):
        return sde.begin(noisy_states,S_sde,X_GT,mask)
    
    
    def test(self, sde=None, save_states=False, save_dir='save_dir', GT = None, mask = None, S_sde = None, S_GT = None, S_LQ = None,dis = None):
        sde.set_mu(self.condition)
        S_sde.set_mu(self.S_LQ)
        self.model.eval()
        self.models.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states, save_dir=save_dir, GT = GT, mask = mask, S_sde = S_sde, S_GT = S_GT, S_LQ = S_LQ, dis = dis, S_LQs = self.S_LQ)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
#         out_dict["Output1"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        load_path_Gs = self.opt["path"]["pretrain_model_Gs"]
        load_path_Gss = self.opt["path"]["pretrain_model_D"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])
            self.load_network(load_path_Gs, self.models, self.opt["path"]["strict_load"])
            self.load_network(load_path_Gss, self.dis, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
        
