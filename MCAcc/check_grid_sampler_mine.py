import torch
import torch.nn.functional as F
from grid_sampler_mine import GridSamplerMine3dFunction,GridSamplerMine3dBackwardFunction
device=torch.device(3)
input=torch.randn(1,5,15,15,15,dtype=torch.double,requires_grad=True).to(device)
grid=(torch.rand(1,1,1,10,3,dtype=torch.double,requires_grad=True).to(device)-0.5)*2.2

GridSamplerMine3dFunction.apply(input,grid)
F.grid_sample(input,grid,mode='bilinear', padding_mode='border', align_corners=False)

torch.autograd.gradcheck(GridSamplerMine3dFunction.apply,(input,grid))


grad_output=torch.randn(1,5,1,1,10,dtype=torch.double,device=device,requires_grad=True)

torch.autograd.gradcheck(GridSamplerMine3dBackwardFunction.apply,(input,grid,grad_output))