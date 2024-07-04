import torch
from torch import jit
from unet import UNet
from torch.utils.mobile_optimizer import optimize_for_mobile
net = UNet(n_channels=3, n_classes=2, bilinear=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
state_dict = torch.load(r"unet_carvana_scale0.5_epoch2.pth", map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

example = torch.rand(1, 3, 240, 320)    

traced_script_module = torch.jit.trace(net, example)
traced_script_module.save("traced_unet.pt")

export_model_name = "unet_torchscript.ptl"

model = torch.jit.load('traced_unet.pt')
optimized_model = optimize_for_mobile(model)
optimized_model._save_for_lite_interpreter(export_model_name)
