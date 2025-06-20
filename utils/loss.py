import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import os
import pyiqa

######### EXPONENTIAL MOVING AVERAGE (EMA) ##########

class ExponentialMovingAverage:
    """Mantiene un registro de los parámetros suavizados de un modelo usando EMA."""

    def __init__(self, model, decay=0.999):
        """
        Args:
            model (torch.nn.Module): Modelo de PyTorch.
            decay (float): Factor de decaimiento de EMA (normalmente 0.999).
        """
        self.model = model
        self.decay = decay
        self.shadow = {}  # Diccionario donde se guardan los valores EMA
        self.backup = {}  # Para restaurar pesos originales

        # Clonar y guardar los parámetros del modelo inicial
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self):
        """Actualiza los pesos EMA en cada paso de entrenamiento."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self):
        """Reemplaza los pesos del modelo con los pesos EMA (para inferencia)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()  # Guarda los pesos actuales
                param.data.copy_(self.shadow[name])  # Reemplaza con EMA

    def restore(self):
        """Restaura los pesos originales después de usar los valores EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])  # Restaura pesos originales
        self.backup = {}  # Limpia el diccionario de backup    

######### GAN LOSS ##########

class GANLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=0.9, fake_label_val=0.1, loss_weight=1.0):
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
                    pred_i = pred_i[-1]  # Tomar solo la última capa si es multiescala
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
        self.multiscaleganloss = MultiScaleGANLoss()

    def forward(self, x_hat, x):

        #print(x.shape)
        #print(x_hat.shape)

        real_d_pred = [
            self.discriminator(x)['out'],                                   # Escala 1x (resolución completa)
            self.discriminator(F.interpolate(x, scale_factor=0.5))['out'],  # Escala 0.5x (mitad)
            self.discriminator(F.interpolate(x, scale_factor=0.25))['out']  # Escala 0.25x (cuarto)
        ]

        fake_d_pred = [
            self.discriminator(x_hat)['out'],                                   # Escala 1x (resolución completa)
            self.discriminator(F.interpolate(x_hat, scale_factor=0.5))['out'],  # Escala 0.5x (mitad)
            self.discriminator(F.interpolate(x_hat, scale_factor=0.25))['out']  # Escala 0.25x (cuarto)
        ]

        # Calcular la pérdida Multi-Scale GAN
        loss_d_real = self.multiscaleganloss(real_d_pred, True)
        loss_d_fake = self.multiscaleganloss(fake_d_pred, False)
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        return loss_d, loss_d_fake, loss_d_real
                
######### GENERATOR LOSS ##########

class LPIPS(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

        # Instanciamos una sola vez la métrica
        self.metric = pyiqa.create_metric('lpips').eval().to('cuda')

    @staticmethod
    def normalize_for_pyiqa(x):
        # Asegura que esté en [0, 1]
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2  # Asume que venía en [-1, 1]
        return x.clamp(0, 1)

    def forward(self, x_hat, x):
        x_hat = self.normalize_for_pyiqa(x_hat)
        x = self.normalize_for_pyiqa(x)

        return self.loss_weight * self.metric(x_hat, x)

class GeneratorLoss(nn.Module):
    def __init__(self, lpips_weight=1.0, gan_weight_max=1.0, k=0.1, t0=500):
        super(GeneratorLoss, self).__init__()
        self.lpips_loss = LPIPS(loss_weight=lpips_weight)
        self.gan_loss = GANLoss(loss_weight=1.0)  # Ponemos 1.0 y escalamos después
        self.l1_loss = nn.L1Loss()

        self.gan_weight_max = gan_weight_max
        self.k = k
        self.t0 = t0

    def sigmoid_weight(self, t):
        # t: step (iteración) actual (int o float)
        return 1 / (1 + math.exp(-self.k * (t - self.t0)))

    def forward(self, x_hat, x, fake_g_pred, step):
        l1_loss = self.l1_loss(x_hat, x)
        gan_weight = self.gan_weight_max * self.sigmoid_weight(step)
        gan_loss = self.gan_loss(fake_g_pred, True) * gan_weight
        lpips_loss = self.lpips_loss(x_hat, x) # si quieres mantener peso fijo

        total_loss = l1_loss + gan_loss + lpips_loss.mean()
        return total_loss, {
            'l1': l1_loss.item(),
            'gan': gan_loss.item(),
            'lpips': lpips_loss.mean().item(),
            'gan_weight': gan_weight
        }



  

