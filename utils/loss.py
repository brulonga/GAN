import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import os
import pyiqa

######### EXPONENTIAL MOVING AVERAGE (EMA) ##########

class ExponentialMovingAverage:

    def __init__(self, model, decay=0.999):

        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {} 

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()  
                param.data.copy_(self.shadow[name])  

    def restore(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name]) 
        self.backup = {}    

######### GAN LOSS ##########

class GANLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        self.loss_fn = nn.BCEWithLogitsLoss() 

    def forward(self, pred, target_is_real):
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        target_tensor = torch.full_like(pred, target_val) 
        return self.loss_weight * self.loss_fn(pred, target_tensor)
    
class MultiScaleGANLoss(GANLoss):
    def forward(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1] 
                loss_tensor = super().forward(pred_i, target_is_real).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real)
        
######### DISCRIMINATOR LOSS ##########
        
class DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator):
        super(DiscriminatorLoss, self).__init__()

        self.discriminator = discriminator
        self.ganloss = GANLoss()

    def forward(self, fake_d_pred, x):

        real_d_pred = self.discriminator(x)['out']

        loss_d_real = self.ganloss(real_d_pred, True)
        loss_d_fake = self.ganloss(fake_d_pred, False)
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        return loss_d, loss_d_fake, loss_d_real
                
######### GENERATOR LOSS ##########

class LPIPS(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.metric = pyiqa.create_metric('lpips', pretrained_model_path='model_zoo/LPIPS_v0.1_alex-df73285e.pth').eval().to('cuda')

    @staticmethod
    def normalize_for_pyiqa(x):
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2
            
        return x.clamp(1e-3, 1-1e-3)

    def forward(self, x_hat, x):
        x_hat = self.normalize_for_pyiqa(x_hat)
        x = self.normalize_for_pyiqa(x)

        return self.loss_weight * self.metric(x_hat, x)

class GeneratorLoss(nn.Module):
    def __init__(self, lpips_weight=1.0, gan_weight_max=1.0, start_epoch= 0, end_epoch=500):
        super(GeneratorLoss, self).__init__()

        self.lpips_loss = LPIPS(loss_weight=lpips_weight)
        self.gan_loss = GANLoss(loss_weight=1.0) 
        self.l1_loss = nn.L1Loss()

        self.gan_weight_max = gan_weight_max
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def sigmoid_weight(self, epoch):
        t0 = (self.start_epoch + self.end_epoch) / 2
        duration = self.end_epoch - self.start_epoch
        k = 8 / duration

        if epoch < self.start_epoch:
            return 0.0
        elif epoch > self.end_epoch:
            return 1.0
        else:
            weight = 1 / (1 + math.exp(-k * (epoch - t0)))
            return weight

    def forward(self, x_hat, x, fake_g_pred, step, annealing = True):

        l1_loss = self.l1_loss(x_hat, x)
        lpips_loss = self.lpips_loss(x_hat, x)

        if annealing:
            gan_weight = self.gan_weight_max * self.sigmoid_weight(step)
        else:
            gan_weight = self.gan_weight_max

        gan_loss = self.gan_loss(fake_g_pred, True) * gan_weight
        
        total_loss = l1_loss + lpips_loss + gan_loss 

        return total_loss, {
            'l1': l1_loss.item(),
            'lpips': lpips_loss.mean().item(),
            'gan': gan_loss.item(),
            'gan_weight': gan_weight
        }



  

