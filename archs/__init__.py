import importlib
import torch
import sys
from utils.utils import count_params, load_weights
from fvcore.nn import FlopCountAnalysis

sys.path.append("./archs/")

def make_model (cfg, device):

    module_name = f"archs.{cfg['arch']}" 
    model_name  = cfg["model"]
    
    module = importlib.import_module(module_name)
    
    model_class = getattr(module, model_name)
    model = model_class(cfg)

    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_fake = torch.rand(1, 3, 960, 540).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()

    nparams_hot = count_params (model.eval())

    print (f"Trainable Parameters {nparams_hot / 1e6} M")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")

    if cfg["ckpt"]:
        try:
            weights = torch.load(cfg["ckpt"], map_location='cpu')
            if 'params_ema' in weights:
                weights = weights['params_ema']
            model = load_weights(model, weights)
        except:
            print ("Failed to load generator", cfg["ckpt"])

    return model

def make_discriminator (cfg, device):

    module_name = f"archs.{cfg['arch']}"  
    model_name  = cfg["model"]
    
    module = importlib.import_module(module_name)

    model_class = getattr(module, model_name)
    model = model_class(cfg)

    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_fake = torch.rand(1, 3, 4*960, 4*540).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()

    nparams_hot   = count_params (model.eval())

    print (f"Trainable Parameters {nparams_hot / 1e6} M")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")

    if cfg["ckpt"]:
        try:
            weights = torch.load(cfg["ckpt"], map_location='cpu')
            if 'params' in weights:
                weights = weights['params']
            model = load_weights(model, weights)
        except:
            print ("Failed to load discriminator", cfg["ckpt"])

    return model