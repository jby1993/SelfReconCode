import torch
from FastMinv import Fast3x3Minv
import cv2
import numpy as np
from time import time

N=10000
device=torch.device(0)
ms=torch.randn(N,3,3).to(device)
ms.mean() # compute to initialize cuda
torch.cuda.synchronize()
start=time()
invs,checks=Fast3x3Minv(ms)
torch.cuda.synchronize()
end=time()
print('%d ms, %d invertible, time:%f'%(checks.numel(),checks.sum().item(),end-start))

errors=(invs[checks].matmul(ms[checks])-torch.eye(3).to(device).view(1,3,3)).norm(dim=(1,2))
print('max:%f, mean:%f.'%(errors.max().item(),errors.mean().item()))


