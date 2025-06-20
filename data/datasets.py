import torch
import torchvision
import os
import random
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.utils import load_img
import utils.degradations as degradation

class SSLSRImage(Dataset):
    """
    Single Image Dataset for Self-Supervised Training of SUPER-RESOLUTION given only high-quality HR images.
    Tasks: synthetic denoising, deblurring, super-res, etc.

    - resize: resize mode ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact') or None
    - size: target size
    """

    def __init__(self, hq_img_paths, size=512, resize="bicubic", logdeg=False, degpipe=0, 
                        scale=2, augmentations=True, sample=-1):

        self.img_paths = hq_img_paths
        self.totensor  = torchvision.transforms.ToTensor()
        self.resize    = resize
        self.size      = size
        self.logdeg    = logdeg
        self.degpipe   = degpipe
        self.augs      = augmentations
        self.scale     = scale

        if sample > 0:
            self.img_paths = self.img_paths[:sample]
            
        self.rcrop  = torchvision.transforms.RandomCrop(size=self.size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = load_img(img_path)
        image = degradation.modcrop(image)
        image = self.totensor(image.astype(np.float32))

        # Random crop the input image
        image = self.rcrop(image)

        # Apply augmentations
        if self.augs:

            # HFLIP
            if torch.rand(1)[0] > 0.5:
                image     = torchvision.transforms.functional.hflip(image)
            
            # VFLIP
            if torch.rand(1)[0] > 0.5:
                image     = torchvision.transforms.functional.vflip(image)

        if self.degpipe == 1:
            deg_image = degradation.degradation_pipeline_1(image, self.scale)
        elif self.degpipe == 2:
            deg_image = degradation.degradation_pipeline_2(image, self.scale)
        else:
            deg_image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1/self.scale, mode="bicubic", antialias=True).squeeze(0)

        assert deg_image.shape[1] == image.shape[1]//self.scale
        assert deg_image.shape[2] == image.shape[2]//self.scale
        assert deg_image.shape[0] == image.shape[0]

        assert not torch.isnan(image).any(), f"NaN in image {img_path}"
        assert not torch.isnan(deg_image).any(), f"NaN in deg_image {img_path}"

        image = torch.clamp(image, 0., 1.)
        deg_image = torch.clamp(deg_image, 0., 1.)
        
        return image, deg_image

class SRSUPDTS(Dataset):
    def __init__(self, name, hr_dir, lr_dir, n, scale):
        """
        Super-Resolution dataset class for evaluation.

        Args:
            name (str): Dataset name.
            hr_dir (str): Path to high-resolution images.
            lr_dir (str): Path to low-resolution images.
            n (int): Expected number of images.
            scale (int): Upscaling factor.
        """
        self.name = name
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.totensor  = torchvision.transforms.ToTensor()

        # Check if directories exist
        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR directory does not exist: {lr_dir}")

        # Get sorted list of image filenames (ensuring matching pairs)
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # Ensure expected number of images
        if len(self.hr_images) != n or len(self.lr_images) != n:
            raise ValueError(f"Dataset '{name}' has {len(self.hr_images)} images, expected {n}.")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        """Returns an LR-HR image pair."""
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_img = load_img(hr_path)
        hr_img = degradation.modcrop(hr_img)
        lr_img = load_img(lr_path)
        hr_img = self.totensor(hr_img.astype(np.float32))
        lr_img = self.totensor(lr_img.astype(np.float32))

        return {"lr": lr_img, "hr": hr_img, "scale": self.scale, "hr_path":hr_path, "lr_path":lr_path}

class SRSSLDTS(Dataset):
    def __init__(self, name, hr_dir, n, scale, mode="bicubic"):
        """
        Super-Resolution dataset class for evaluation.

        Args:
            name (str): Dataset name.
            hr_dir (str): Path to high-resolution images.
            n (int): Expected number of images.
            scale (int): Upscaling factor.
        """
        self.name   = name
        self.hr_dir = hr_dir
        self.scale  = scale
        self.resize = mode
        self.totensor  = torchvision.transforms.ToTensor()

        # Check if directories exist
        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")

        # Get sorted list of image filenames (ensuring matching pairs)
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # Ensure expected number of images
        if len(self.hr_images) != n:
            raise ValueError(f"Dataset '{name}' has {len(self.hr_images)} images, expected {n}.")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        """Returns an LR-HR image pair."""
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_img = load_img(hr_path)
        hr_img = degradation.modcrop(hr_img)
        hr_img = self.totensor(hr_img.astype(np.float32))
        lr_img = torch.nn.functional.interpolate(hr_img.unsqueeze(0), scale_factor=1/self.scale, mode="bicubic", antialias=True).squeeze(0)
        return {"lr": lr_img, "hr": hr_img, "scale": self.scale, "hr_path":hr_path}
    
class SRNRIQADTS(Dataset):
    def __init__(self, name, lr_dir, n):
        """
        Super-Resolution dataset class for evaluation.

        Args:
            name (str): Dataset name.
            hr_dir (str): Path to low-resolution images.
            n (int): number of images we want to process.
        """
        self.name   = name
        self.lr_dir = lr_dir
        self.totensor  = torchvision.transforms.ToTensor()

        # Check if directories exist
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR directory does not exist: {lr_dir}")

        # Get sorted list of image filenames (ensuring matching pairs)
        all_images = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.lr_images = all_images[:n]

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        """Returns an LR-HR image pair."""
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        lr_img = load_img(lr_path)
        lr_img = degradation.modcrop(lr_img)
        lr_img = self.totensor(lr_img.astype(np.float32))
        return {"lr": lr_img}