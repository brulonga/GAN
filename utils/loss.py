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
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_fn = nn.BCEWithLogitsLoss() 

    def forward(self, pred, target_is_real):
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        target_tensor = torch.full_like(pred, target_val) 
        return self.loss_fn(pred, target_tensor)

class RelativisticGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake, for_discriminator=True):
        # Validación: asegurar que las dimensiones coincidan
        assert pred_real.shape == pred_fake.shape, f"Shape mismatch: {pred_real.shape} vs {pred_fake.shape}"

        mean_real = pred_real.mean()
        mean_fake = pred_fake.mean()

        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)

        if for_discriminator:
            # D(x_real) - mean(D(x_fake)) → 1
            # D(x_fake) - mean(D(x_real)) → 0
            loss_real = self.loss_fn(pred_real - mean_fake, real_label)
            loss_fake = self.loss_fn(pred_fake - mean_real, fake_label)
        else:
            # Inverso para el generador
            loss_real = self.loss_fn(pred_real - mean_fake, fake_label)
            loss_fake = self.loss_fn(pred_fake - mean_real, real_label)

        return (loss_real + loss_fake) / 2, loss_real, loss_fake

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
    def __init__(self, discriminator, relativistic=False):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.relativistic = relativistic

        if self.relativistic:
            self.ganloss = RelativisticGANLoss()
        else:
            self.ganloss = GANLoss()

    def forward(self, fake_d_pred, real_d_pred):
        if self.relativistic:
            assert real_d_pred is not None, "real_d_pred must be provided for relativistic GAN in Discriminator loss"
            loss_d, loss_d_real, loss_d_fake = self.ganloss(real_d_pred, fake_d_pred, for_discriminator=True)
        else:
            assert real_d_pred is not None, "real_d_pred must be provided for standard GAN in Discriminator loss"
            loss_d_real = self.ganloss(real_d_pred, True)
            loss_d_fake = self.ganloss(fake_d_pred, False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5

        return loss_d, loss_d_fake, loss_d_real

######### GENERATOR LOSS ##########

class LPIPS(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        if self.loss_weight > 0:
            self.metric = pyiqa.create_metric(
                'lpips',
                pretrained_model_path='model_zoo/LPIPS_v0.1_alex-df73285e.pth'
            ).eval().to('cuda')
        else:
            self.metric = None  # Evita cargar el modelo si no se usará

    @staticmethod
    def normalize_for_pyiqa(x):
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2
        return x.clamp(1e-3, 1 - 1e-3)

    def forward(self, x_hat, x):
        if self.loss_weight == 0.0 or self.metric is None:
            return torch.tensor(0.0, device=x.device)

        x_hat = self.normalize_for_pyiqa(x_hat)
        x = self.normalize_for_pyiqa(x)
        return self.loss_weight * self.metric(x_hat, x)

class GeneratorLoss(nn.Module):
    def __init__(self, l1_weight=1.0, lpips_weight=1.0, gan_weight_max=1.0, start_epoch=0, end_epoch=500, relativistic=False):
        super(GeneratorLoss, self).__init__()

        self.lpips_loss = LPIPS(loss_weight=lpips_weight)
        self.l1_loss = nn.L1Loss()

        self.relativistic = relativistic
        if self.relativistic:
            self.gan_loss = RelativisticGANLoss()
        else:
            self.gan_loss = GANLoss()

        self.gan_weight_max = gan_weight_max
        self.l1_weight = l1_weight
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
            return 1 / (1 + math.exp(-k * (epoch - t0)))

    def forward(self, x_hat, x, fake_d_pred, real_d_pred, step, annealing=True):
        # Reconstrucción
        l1_loss = self.l1_loss(x_hat, x) * self.l1_weight
        lpips_loss = self.lpips_loss(x_hat, x)

        # Peso del adversarial
        gan_weight = self.gan_weight_max * self.sigmoid_weight(step) if annealing else self.gan_weight_max

        # GAN loss
        if self.relativistic:
            assert real_d_pred is not None, "real_d_pred must be provided for relativistic GAN"
            gan_loss_val, _, _ = self.gan_loss(real_d_pred, fake_d_pred, for_discriminator=False)
        else:
            gan_loss_val = self.gan_loss(fake_d_pred, True)

        gan_loss = gan_loss_val * gan_weight

        total_loss = l1_loss + lpips_loss + gan_loss

        return total_loss, {
            'l1': l1_loss.item(),
            'lpips': lpips_loss.mean().item(),
            'gan': gan_loss.item(),
            'gan_weight': gan_weight
        }
