import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_mixed_kernels, circular_lowpass_kernel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def modcrop(img_in, scale=4):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def degradation_pipeline_real_esrgan(img, scale=4,
                                     kernel_size1=21,
                                     kernel_list1=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                                     kernel_prob1=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                                     sigma_range1=(0.2, 3),
                                     betag_range1=(0.5, 4),
                                     betap_range1=(1, 2),
                                     sinc_prob1=0.1,
                                     resize_prob1=[0.2, 0.7, 0.1],
                                     resize_range=(0.15, 1.5),
                                     gaussian_noise_prob=0.5,
                                     noise_range_gauss1=(1, 30),
                                     noise_range_poisson1=(0.05, 3),
                                     gray_noise_prob=0.4,
                                     jpeg_range1=(30, 95),
                                     second_blur_prob=0.8,
                                     kernel_size2=21,
                                     kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                                     kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                                     sigma_range2=(0.2, 3),
                                     betag_range2=(0.5, 4),
                                     betap_range2=(1, 2),
                                     sinc_prob2=0.1,
                                     resize_prob2=[0.2, 0.7, 0.1],
                                     resize_range2=(0.15, 1.5),
                                     gaussian_noise_prob2=0.5,
                                     noise_range_gauss2=(1, 30),
                                     noise_range_poisson2=(0.05, 3),
                                     gray_noise_prob2=0.4,
                                     jpeg_range2=(30, 95),
                                     final_sinc_prob=0.8,
                                     jpeger=None):

    c, h, w = img.size()
    device = img.device
    img = img.unsqueeze(0)

    k = random_mixed_kernels(kernel_list1, kernel_prob1, kernel_size=kernel_size1,
                             sigma_x_range=sigma_range1, sigma_y_range=sigma_range1,
                             rotation_range=(-math.pi, math.pi),
                             betag_range=betag_range1, betap_range=betap_range1)
    kernel_tensor = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
    out = filter2D(img, kernel_tensor)

    if random.random() < sinc_prob1:
        cutoff = np.pi * random.uniform(0.4, 1.0)
        sinc_k = circular_lowpass_kernel(cutoff=cutoff, kernel_size=kernel_size1)
        sinc_k_tensor = torch.tensor(sinc_k, dtype=torch.float32, device=device).unsqueeze(0)
        out = filter2D(out, sinc_k_tensor)

    updown_type = random.choices(['up', 'down', 'keep'], resize_prob1)[0]
    scale_resize = 1.0 if updown_type == 'keep' else (
        random.uniform(*resize_range) if updown_type == 'up' else random.uniform(resize_range[0], 1))
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale_resize, mode=mode,
                        antialias=(mode != 'area'), recompute_scale_factor=True)

    if random.random() < gaussian_noise_prob:
        out = random_add_gaussian_noise_pt(out.squeeze(0), sigma_range=noise_range_gauss1, clip=True,
                                           rounds=False, gray_prob=gray_noise_prob).unsqueeze(0)
    else:
        out = random_add_poisson_noise_pt(out.squeeze(0), scale_range=noise_range_poisson1, clip=True,
                                          rounds=False, gray_prob=gray_noise_prob).unsqueeze(0)

    jpeg_quality_tensor = out.new_tensor([random.uniform(*jpeg_range1)])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_quality_tensor)

    if random.random() < second_blur_prob:
        k2 = random_mixed_kernels(kernel_list2, kernel_prob2, kernel_size=kernel_size2,
                                  sigma_x_range=sigma_range2, sigma_y_range=sigma_range2,
                                  rotation_range=(-math.pi, math.pi),
                                  betag_range=betag_range2, betap_range=betap_range2)
        kernel_tensor2 = torch.tensor(k2, dtype=torch.float32, device=device).unsqueeze(0)
        out = filter2D(out, kernel_tensor2)

    if random.random() < sinc_prob2:
        cutoff = np.pi * random.uniform(0.4, 1.0)
        sinc_k = circular_lowpass_kernel(cutoff=cutoff, kernel_size=kernel_size2)
        sinc_k_tensor = torch.tensor(sinc_k, dtype=torch.float32, device=device).unsqueeze(0)
        out = filter2D(out, sinc_k_tensor)

    updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
    scale_resize = 1.0 if updown_type == 'keep' else (
        random.uniform(*resize_range2) if updown_type == 'up' else random.uniform(resize_range2[0], 1))
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale_resize, mode=mode,
                        antialias=(mode != 'area'), recompute_scale_factor=True)

    if random.random() < gaussian_noise_prob2:
        out = random_add_gaussian_noise_pt(out.squeeze(0), sigma_range=noise_range_gauss2, clip=True,
                                           rounds=False, gray_prob=gray_noise_prob2).unsqueeze(0)
    else:
        out = random_add_poisson_noise_pt(out.squeeze(0), scale_range=noise_range_poisson2, clip=True,
                                          rounds=False, gray_prob=gray_noise_prob2).unsqueeze(0)

    final_size = (h // scale, w // scale)
    apply_sinc = random.random() < final_sinc_prob
    jpeg_quality_tensor = out.new_tensor([random.uniform(*jpeg_range2)])

    if apply_sinc:
        cutoff = np.pi * random.uniform(0.4, 1.0)
        sinc_kernel = circular_lowpass_kernel(cutoff=cutoff, kernel_size=21)
        sinc_kernel = torch.tensor(sinc_kernel, dtype=torch.float32, device=img.device).unsqueeze(0)

    if random.random() < 0.5:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=final_size, mode=mode,
                            align_corners=False if mode != 'area' else None)
        if apply_sinc:
            out = filter2D(out, sinc_kernel)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_quality_tensor)
    else:
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_quality_tensor)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=final_size, mode=mode,
                            align_corners=False if mode != 'area' else None)
        if apply_sinc:
            out = filter2D(out, sinc_kernel)

    return torch.clamp(out, 0, 1).squeeze(0)