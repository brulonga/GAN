import numpy as np
import os
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader
import wandb
import pyiqa

from utils import *
from utils.metrics import pt_psnr, get_psnry, get_ssimy
from pytorch_msssim import ssim
from utils.loss import GeneratorLoss, DiscriminatorLoss, ExponentialMovingAverage

def test_model(model, testsets, device, use_wandb, ema):

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
                #psnr_y = [] ; ssim_y = []

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
                    #psnr_y.append (get_psnry(x, x_hat).item())
                    #ssim_y.append (get_ssimy(x, x_hat).item())

                    if idx==0:
                        # Plot only the first image of each test set
                        bi = 0
                        _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                        _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                        _x    = np.clip(x[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                        if use_wandb:
                            wandb.log({f"{testset_name}_results": [wandb.Image(image) for image in [_y, _x_hat, _x]]})

                        del _x, _y,_x_hat; gc.collect()

                ## Log the test metrics:
                if use_wandb:
                    wandb.log({f"{testset_name}_{testset.scale}_psnr": np.mean(psnr_rgb)})
                    wandb.log({f"{testset_name}_{testset.scale}_ssim": np.mean(ssim_rgb)})
                    #wandb.log({f"{testset_name}_{testset.scale}_PSNR-Y": np.mean(psnr_y)})
                    #wandb.log({f"{testset_name}_{testset.scale}_SSIM-Y": np.mean(ssim_y)})

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
                            # Plot only the first image of each test set
                            bi = 0
                            _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                            _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                            if use_wandb:
                                wandb.log({f"{testset_name}_results": [wandb.Image(image) for image in [_y, _x_hat]]})

                            del _y,_x_hat; gc.collect()

                ## Log the test metrics:
                if use_wandb:
                    wandb.log({f"{testset_name}_niqe": np.mean(niqe_list)})

                del test_dataloader; gc.collect()

    ema.restore()

def fit_sr (generator, discriminator, optimizerG, optimizerD, dataloader, device, testsets=None, use_wandb=True, use_amp=True, epochs=100, 
                  verbose=10, modelname="testmodel", out_path="results/", start_epoch=0, ema_flag = True, clip_value = 1.0, lpips_weight = 1.0, 
                  gan_weight_max = 1.0, k = 0.1, t0= 500):

    steps=0
    use_amp=use_amp
    nan_batches = 0  # Track NaN batches

    ema = ExponentialMovingAverage(generator, decay=0.999)

    loss_g = GeneratorLoss(lpips_weight= lpips_weight, gan_weight_max=gan_weight_max, k=k, t0=t0)
    loss_d = DiscriminatorLoss(discriminator=discriminator)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #By the moment no lr scheduler
    #scheduler =torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.004, epochs=100, div_factor=10, final_div_factor=4000)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in tqdm(range(start_epoch, start_epoch+epochs), total=start_epoch+epochs):

        generator.train()
        discriminator.train()

        for batch in dataloader:

            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            ########## GENERAR x_hat ##########
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                x_hat = generator(y)['img']

            # Detectar NaNs/Infs en la salida antes de cualquier cálculo de loss
            if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
                print(f"x_hat contiene NaN o Inf en step {steps}, batch saltado")
                nan_batches += 1
                continue  # saltar esta iteración completa (no update D ni G)

            ########## DISCRIMINATOR UPDATE ##########
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                for param in discriminator.parameters():
                    param.requires_grad = True

                optimizerD.zero_grad(set_to_none=True)

                x_hat_detached = x_hat.detach()

                loss_disc, loss_d_fake, loss_d_real = loss_d(x_hat_detached, x)

            # prev_params = {}

            # for name, param in discriminator.named_parameters():
            #     if param.requires_grad:
            #         prev_params[name] = param.clone().detach()           

            if use_amp:

                scaler.scale(loss_disc).backward()
                torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value)  
                scaler.step(optimizerD)  # Update discriminator weights
                scaler.update()  # Update the scaler

            else:

                loss_disc.backward()

                # def print_gradients(model):
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             grad_norm = param.grad.data.norm(2)
                #             if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                #                 print(f"[!] {name}: gradiente NaN o Inf detectado")
                #             else:
                #                 print(f"{name}: grad norm = {grad_norm:.4f}")
                #         else:
                #             print(f"{name}: No tiene gradiente")

                # # Ejemplo, después de loss.backward():

                # print_gradients(discriminator)

                torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value)
                optimizerD.step()  

            # for name, param in discriminator.named_parameters():
            #     if param.requires_grad:
            #         # Calcular la diferencia entre el parámetro anterior y el actual
            #         diff = (param - prev_params[name]).abs().sum()
            #         print(f"{name}: cambio total = {diff.item():.6f}")

            ########## GENERATOR UPDATE ##########

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                
                for param in discriminator.parameters():
                    param.requires_grad = False

                optimizerG.zero_grad(set_to_none=True)

                fake_pred_disc = discriminator(x_hat)['out']
                loss_gen, dict_losses = loss_g(x_hat, x, fake_pred_disc, epoch)
                            
                #psnr_loss = torch.mean(pt_psnr(x, x_hat))
                #ssim_loss = ssim(x, x_hat, data_range=1., size_average=True)
                #psnry_loss= get_psnry(x, x_hat)
                #ssimy_loss= get_ssimy(x, x_hat)
                
            if use_amp:

                scaler.scale(loss_gen).backward() 
                torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value)
                scaler.step(optimizerG)  # Update generator weights
                scaler.update()  # Update the scaler for the next iteration

                if ema_flag:
                    ema.update()

            else:

                loss_gen.backward()
                torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value)
                optimizerG.step()  

                if ema_flag:
                    ema.update()

            steps+=1
            if use_wandb:
                #wandb.log({"psnr": psnr_loss.item()})
                wandb.log({"lossG": loss_gen.item()})
                wandb.log({"lossD": loss_disc.item()})
                #wandb.log({"ssim": ssim_loss.item()})
                wandb.log({"LPIPS": dict_losses['lpips']})
                wandb.log({"GANLoss": dict_losses['gan']})
                wandb.log({"l1": dict_losses['l1']})
                #wandb.log({"psnr-y": psnry_loss.item()})
                #wandb.log({"ssim-y": ssimy_loss.item()})
                wandb.log({"Fake_disc_loss": loss_d_fake.item()})
                wandb.log({"Real_disc_loss": loss_d_real.item()})
            
        #scheduler.step()

        if nan_batches >= 10: 
            print ("Training collapses")
            break

        if (epoch % verbose)==0:
            
            print (f"\n >> Epoch {epoch} / {start_epoch+epochs}", 'LossG', loss_gen.item(), 'LossD', loss_disc.item(), "Steps", steps)
            print ("model saved at ", os.path.join(out_path, f"{modelname}.pt"))

            for bi in range(3):
                _y    = np.clip(y[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                _x_hat= np.clip(x_hat[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                _x    = np.clip(x[bi].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)

                if use_wandb:
                    wandb.log({f"train_images_{bi}": [wandb.Image(image) for image in [_y, _x_hat, _x]]})

                del _x, _y,_x_hat; gc.collect()
            
            print('Epoca completada')

            ##### Evaluate model
            test_model(generator, testsets, device, use_wandb, ema)

            ##### Save Model
            torch.save(generator.state_dict(), os.path.join(out_path, f"{modelname}_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(out_path, f"{modelname}_{epoch}_discriminator.pt"))
            
            gc.collect()
        
        #torch.save(model.state_dict(), os.path.join(out_path, f"{modelname}_{epoch}.pt"))

    # torch.save(generator.state_dict(), os.path.join(out_path, f"{modelname}_last.pt"))
    # torch.save(discriminator.state_dict(), os.path.join(out_path, f"{modelname}_discriminator_last.pt"))

                # Handle NaN or Inf Loss
                # if torch.isnan(loss_d) or torch.isinf(loss_d):
                #     nan_batches += 1  # Track NaN batches
                #     print(f"⚠️ Skipping step due to unstable discriminator loss: {loss_d.item()}")
                #     continue 

                # # Handle NaN or Inf Loss
                # if torch.isnan(loss_g) or torch.isinf(loss_g):
                #     nan_batches += 1  # Track NaN batches
                #     print(f"⚠️ Skipping step due to unstable generator loss: {loss_g.item()}")
                #     continue 

                # for name, param in generator.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                #             print(f"⚠️ NaN or Inf detected in gradient of {name}!")
                #             continue

                # for name, param in discriminator.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                #             print(f"⚠️ NaN or Inf detected in gradient of {name}!")
                #             continue