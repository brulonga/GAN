import torch
import random
import torch.nn.functional as F
import numpy as np

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

########## TORCH DEGRADATIONS ##########

########## DOWNSAMPLING ##########

def downsampling(image, scale):

    mode = random.choice(['nearest', 'bilinear', 'bicubic', 'area'])

    if mode in ['bilinear', 'bicubic']:
        deg_image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode=mode, antialias=True).squeeze(0)
    else:
        deg_image = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode=mode, antialias=False).squeeze(0)

    return deg_image

########## NOISE ##########

def add_gaussian_noise(image, scale):

    mean = 0.0
    std = random.uniform(0.01, 0.1)

    noise = torch.randn_like(image) * std + mean
    noisy_img = image + noise

    return torch.clamp(noisy_img, 0.0, 1.0)

########## JPEG COMPRESSION SIMULATION (GPU) ##########

def simulate_blocking(img, block_size=8, levels=32):
    C, H, W = img.shape
    H_crop, W_crop = H - H % block_size, W - W % block_size
    img = img[:, :H_crop, :W_crop]
    
    img_blocks = img.unfold(1, block_size, block_size).unfold(2, block_size, block_size)  # (C, H//B, W//B, B, B)
    img_blocks = torch.round(img_blocks * levels) / levels
    img = img_blocks.contiguous().permute(0, 1, 3, 2, 4).reshape(C, H_crop, W_crop)
    
    return img

def simulate_color_quantization(img, levels=32):
    return torch.round(img * levels) / levels


def fast_fake_jpeg(img, scale):

    if random.random() < 0.7:
        img = simulate_blocking(img, block_size=8, levels=random.randint(16, 64))
    if random.random() < 0.7:
        img = F.avg_pool2d(img.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    if random.random() < 0.7:
        img = simulate_color_quantization(img, levels=random.randint(16, 64))

    return img

########## BLUR ##########

def get_gaussian_kernel(kernel_size, sigma, device):
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma)**2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d

def get_avg_kernel(kernel_size, device):
    kernel = torch.ones((kernel_size, kernel_size), device=device)
    kernel /= kernel.sum()
    return kernel

def get_motion_kernel(kernel_size, direction='horizontal', device='cpu'):
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    if direction == 'horizontal':
        kernel[kernel_size // 2, :] = 1
    else:
        kernel[:, kernel_size // 2] = 1
    kernel /= kernel.sum()
    return kernel

def apply_blur(img, kernel):
    if img.dim() != 3:
        raise ValueError("Esperado tensor [C, H, W]")
    c = img.shape[0]
    kernel = kernel.expand(c, 1, *kernel.shape)  # [C,1,k,k]
    return F.conv2d(img.unsqueeze(0), kernel, padding=kernel.shape[-1] // 2, groups=c).squeeze(0)

def random_blur(img, scale):
    device = img.device
    kernel_size = random.choice([3, 5, 7])
    blur_type = random.choice(['gaussian', 'average', 'motion'])
    
    if blur_type == 'gaussian':
        sigma = random.uniform(0.3, 1.0)
        kernel = get_gaussian_kernel(kernel_size, sigma, device)
    elif blur_type == 'average':
        kernel = get_avg_kernel(kernel_size, device)
    elif blur_type == 'motion':
        direction = random.choice(['horizontal', 'vertical'])
        kernel = get_motion_kernel(kernel_size, direction, device)
    return apply_blur(img, kernel)


########## PIPELINE ##########

def degradation_pipeline_1(image, scale):

    if image.dim() != 3:
        raise ValueError(f'Esperaba tensor 3D [C, H, W], pero recibí {image.shape}')

    deg2function = {'downsample': downsampling, 'noise': add_gaussian_noise, 'jpeg': fast_fake_jpeg, 'none': lambda x, s: x}

    degs = ['noise', 'jpeg', 'none']

    deg1 = random.choice(degs)

    degs = ['downsample'] + [deg1]

    np.random.shuffle(degs)

    for d in degs:

        image = deg2function[d](image, scale)

    return image

def degradation_pipeline_2(image, scale):

    if image.dim() != 3:
        raise ValueError(f'Esperaba tensor 3D [C, H, W], pero recibí {image.shape}')

    deg2function = {'downsample': downsampling, 'noise': add_gaussian_noise, 'jpeg': fast_fake_jpeg, 'blur': random_blur, 'none': lambda x, s: x}

    degs = ['noise', 'jpeg', 'blur', 'none']

    deg1, deg2 = random.sample(degs, 2)

    degs = ['downsample'] + [deg1] + [deg2]

    np.random.shuffle(degs)

    for d in degs:
        image = deg2function[d](image, scale)

    return image

