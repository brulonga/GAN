import torch
from torch.utils.data import DataLoader
import os
import yaml
from glob import glob
import wandb
import argparse
from datetime import datetime
import numpy as np
import random


from utils.utils import dict2namespace, seed_everything
from data.datasets import SSLSRImage, SRSSLDTS, SRSUPDTS, SRNRIQADTS
from scripts.train import fit_sr
from scripts.train_l2 import fit_sr_l2
from basicsr.utils import DiffJPEG, USMSharp

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg/base.yml', help='Path to config file')
    parser.add_argument('--test', type=str, default='test.yml', help='Path to Test config file')
    parser.add_argument('--device', type=int, default=0, help="GPU device")
    parser.add_argument('--debug',  action='store_true', help="Debug mode")
    parser.add_argument('--name', type=str, default='test-model')
    args = parser.parse_args()

    SEED=42
    seed_everything(SEED=SEED)
    torch.backends.cudnn.deterministic = True

    GPU        = args.device
    DEBUG      = args.debug
    MODEL_NAME = args.name
    CONFIG     = args.config
    TEST_CFG   = args.test
    
    MODEL_NAME += "_" + datetime.now().strftime("%d%H%M")

    torch.cuda.set_device(f'cuda:{GPU}')
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)

    with open(os.path.join("config/degradations.yml"), "r") as f:
        config_deg = yaml.safe_load(f)  

    cfg_deg = dict2namespace(config_deg) 

    ################### WANDB LOG

    USE_WANDB   = cfg.wandb.use
    if USE_WANDB and (not DEBUG):
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=MODEL_NAME, config=cfg, tags=cfg.wandb.tags)
        wandb.save("datasets.py")
        wandb.save(f"{CONFIG}")

    print (20*"****")
    print (MODEL_NAME, device, DEBUG, USE_WANDB, CONFIG)
    print (20*"****")

    ################### TRAINING DATASET

    SSL_DATASET = []

    for idx_dts, dts in enumerate(cfg.data.ssl_datasets):

        if "LSDIR" in dts:
            img_list = glob(os.path.join(dts, "*/*.png"))#[:cfg.data.ssl_samples[idx_dts]]
        else:
            img_list = glob(os.path.join(dts, "*.png"))#[:cfg.data.ssl_samples[idx_dts]]
        
        print (dts, len(img_list))
        SSL_DATASET += img_list
        del img_list

    print ("All images SSL", len(SSL_DATASET))

    jpeger = DiffJPEG(differentiable=False)
    usm_sharp = USMSharp()

    dataset    = SSLSRImage(hq_img_paths=SSL_DATASET,
                         resize='bicubic', size=cfg.data.crop_size, scale=cfg.data.scale,
                         degpipe=None, augmentations=True, logdeg=False, cfg_deg=cfg_deg, jpeger=jpeger,
                         sharp=cfg.training.sharp, usm_sharp=usm_sharp)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers, 
                            drop_last=True, pin_memory=True, shuffle=True, worker_init_fn=seed_worker,
                            generator=g)
    
    ################### TESTING DATASET

    with open(TEST_CFG, "r") as file:
        config = yaml.safe_load(file)

    test_datasets = []
    for dataset_cfg in config["test"]:
        for name, params in dataset_cfg.items():
            if name == 'DIV2K_LSDIR':
                try:
                    ts_dataset = SRSUPDTS(
                        name=name,
                        hr_dir=params["hr"],
                        lr_dir=params["lr"],
                        n=params["n"],
                        scale=params["scale"],
                    )
                    print ("Added Supervised Test set:", name)

                except KeyError:
                    ts_dataset = SRSSLDTS(
                        name=name,
                        hr_dir=params["hr"],
                        n=params["n"],
                        scale=params["scale"],
                    )
                    print ("Added Bic Test set:", name)
            
            else:
                ts_dataset = SRNRIQADTS(
                    name=name,
                    lr_dir=params["lr"],
                    n = params["n"]
                )
                print ("Added NRIQA dataset:", name)
                
            test_datasets.append(ts_dataset)

    print ("Testsets", len(test_datasets))

    ################### MODEL

    USE_WANDB = cfg.wandb.use

    from archs.__init__ import make_model, make_discriminator

    print('Generator:')

    model_config = cfg.generator.models[cfg.generator.pick]
    model = make_model (model_config, device)

    print('Discriminator:')

    discriminator_config = cfg.discriminator.models[cfg.discriminator.pick]
    discriminator = make_discriminator(discriminator_config, device)

    optimizerD = torch.optim.AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=cfg.optim.lr_discriminator, weight_decay=cfg.optim.weight_decay)
    optimizerG = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=cfg.optim.T_max ,eta_min=cfg.optim.eta_min)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"""
        Experiment {MODEL_NAME}
        ---------------------------------
        Dataset {cfg.data.ssl_datasets}
        Images  {dataset.__len__()}
        Batch size {cfg.training.batch_size}
        ---------------------------------
        """)
    
    print (20*"****")
    ################### TRAINING

    if not DEBUG:

        if USE_WANDB:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=MODEL_NAME, config=cfg, tags=cfg.wandb.tags)

        if cfg.training.only_l2:

            fit_sr_l2(model, optimizerG, schedulerG, dataloader, device, test_datasets, use_wandb=USE_WANDB, use_amp=cfg.optim.amsgrad, epochs=cfg.training.epochs,
                    verbose=cfg.training.log_freq, modelname=MODEL_NAME, out_path=f"./results/{MODEL_NAME}/", clip_value=cfg.optim.grad_clip,
                    )

        else:

            fit_sr (model, discriminator, optimizerG, optimizerD, dataloader, testsets=test_datasets, device=device, use_wandb=USE_WANDB, 
                epochs=cfg.training.epochs, verbose=cfg.training.log_freq, use_amp=cfg.optim.amsgrad,
                modelname=MODEL_NAME, out_path=f"./results/{MODEL_NAME}/", clip_value = cfg.optim.grad_clip, lpips_weight=cfg.training.lpips_weight,
                gan_weight_max=cfg.training.gan_weight_max, schedulerG = schedulerG, start_epoch=cfg.training.start_epoch, end_epoch= cfg.training.end_epoch,
                annealing=cfg.training.annealing, ema_flag=cfg.training.ema_flag,
                )

        if USE_WANDB:
            wandb.finish()

    print (" TO BE CONTINUED ")
    print (20*"****")   