import numpy as np
import os
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader
import wandb
import pyiqa
import torch.nn.functional as F

from utils import *
from utils.metrics import pt_psnr, get_psnry, get_ssimy
from pytorch_msssim import ssim
from utils.loss import GeneratorLoss, DiscriminatorLoss, ExponentialMovingAverage

def test_model_l2(model, testsets, device, use_wandb, ema):

    print ("Testing model ...")
    model.eval()

    ema.apply_shadow()

    with torch.no_grad():

        for testset in testsets:
            print (">>> ", testset.name)
            testset_name = testset.name
            test_dataloader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

            if testset_name == 'DIV2K_LSDIR':

                psnr_rgb   = [] ; ssim_rgb = []

                for idx, batch in enumerate(test_dataloader):

                    x = batch["hr"].to(device)
                    try:
                        y = batch["lr"].to(device)
                    except:
                        y = torch.nn.functional.interpolate(x.unsqueeze(0), scale_factor=1/testset.scale, mode=testset.resize).squeeze(0)

                    x_hat = model(y)["img"]

                    assert x_hat.shape == x.shape, f"Shape Problem, sr/hr {x_hat.shape,x.shape}"

                    psnr_rgb.append(torch.mean(pt_psnr(x, x_hat)).item())
                    ssim_rgb.append(ssim(x, x_hat, data_range=1., size_average=True).item())

                    if idx==0:

                        bi = 0
                        _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                        _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                        _x    = np.clip(x[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                        if use_wandb:
                            wandb.log({f"{testset_name}_results": [wandb.Image(image) for image in [_y, _x_hat, _x]]})

                        del _x, _y,_x_hat; gc.collect()


                if use_wandb:
                    wandb.log({f"{testset_name}_{testset.scale}_psnr": np.mean(psnr_rgb)})
                    wandb.log({f"{testset_name}_{testset.scale}_ssim": np.mean(ssim_rgb)})

                del test_dataloader; gc.collect()

            else:
                
                niqe_list = []
                niqe_model = pyiqa.create_metric('niqe').eval().to(device)

                for idx, batch in enumerate(test_dataloader):

                    y = batch["lr"].to(device)
                    x_hat = model(y)["img"]

                    niqe =  niqe_model(x_hat).item()
                    niqe_list.append(niqe)

                    if idx==0:
                            
                            bi = 0
                            _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                            _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                            if use_wandb:
                                wandb.log({f"{testset_name}_results": [wandb.Image(image) for image in [_y, _x_hat]]})

                            del _y,_x_hat; gc.collect()

                if use_wandb:
                    wandb.log({f"{testset_name}_niqe": np.mean(niqe_list)})

                del test_dataloader; gc.collect()

    ema.restore()

def fit_sr_l2(generator, optimizerG, scheduler, dataloader, device, testsets=None, use_wandb=True, use_amp=True, epochs=100, 
                  verbose=10, modelname="testmodel", out_path="results/", start_epoch=0, ema_flag = True, 
                  clip_value = 1.0):

    steps=0
    use_amp=use_amp
    nan_batches = 0

    ema = ExponentialMovingAverage(generator, decay=0.999)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in tqdm(range(start_epoch, start_epoch+epochs), total=start_epoch+epochs):

        generator.train()

        for batch in dataloader:

            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                x_hat = generator(y)['img']

            if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
                print(f"x_hat had NaN o Inf in step {steps}, skipt batch")
                nan_batches += 1
                continue  

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):

                optimizerG.zero_grad(set_to_none=True)

                loss = F.mse_loss(x_hat, x)
                psnr_loss = torch.mean(pt_psnr(x, x_hat))
                ssim_loss = ssim(x, x_hat, data_range=1., size_average=True)
                
            if use_amp:

                scaler.scale(loss).backward()
                scaler.unscale_(optimizerG)
                torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value)
                scaler.step(optimizerG)  
                scaler.update() 

                if ema_flag:
                    ema.update()

            else:

                loss.backward()
                torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value)
                optimizerG.step()  

                if ema_flag:
                    ema.update()

            steps+=1
            if use_wandb:
                wandb.log({"l2": loss.item()})
                wandb.log({"psnr": psnr_loss.item()})
                wandb.log({"ssim": ssim_loss.item()})
            
        scheduler.step()

        if nan_batches >= 10: 
            print ("Training collapses")
            break

        if (epoch % verbose)==0:
            
            print (f"\n >> Epoch {epoch} / {start_epoch+epochs}", 'Loss', loss.item(), "Steps", steps)
            print ("model saved at ", os.path.join(out_path, f"{modelname}.pt"))

            for bi in range(3):
                _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                _x    = np.clip(x[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                if use_wandb:
                    wandb.log({f"train_images_{bi}": [wandb.Image(image) for image in [_y, _x_hat, _x]]})

                del _x, _y,_x_hat; gc.collect()
            
            print('Completed epoch')
            test_model_l2(generator, testsets, device, use_wandb, ema)
            torch.save(generator.state_dict(), os.path.join(out_path, f"{modelname}.pt"))
            
            gc.collect()