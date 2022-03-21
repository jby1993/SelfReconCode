import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import GridSamplerMine

class GridSamplerMine3dFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='border', align_corners=False):
        ctx.save_for_backward(input,grid)
        if align_corners==True:
            raise NotImplementedError
        # if torch.isnan(input).any() or torch.isnan(grid).any():
        #     print('forward0')
        #     assert(False)
        out=GridSamplerMine.forward(input,grid,0,1)
        # if torch.isnan(out).any():
        #     print('forward1 ')
        #     assert(False)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        # if torch.isnan(input).any():
        #     print('backward0_0')
        #     assert(False)
        # if torch.isnan(grid).any():
        #     print('backward0_1')
        #     assert(False)
        # if torch.isnan(grad_output).any():
        #     print('backward0_2')
        #     assert(False)
        o0,o1=GridSamplerMine3dBackwardFunction.apply(input,grid,grad_output)
        # if torch.isnan(o0).any():
        #     print('backward1_0')
        #     assert(False)
        # if torch.isnan(o1).any():
        #     print('backward1_1')
        #     assert(False)
        return o0,o1

class GridSamplerMine3dBackwardFunction(Function):
    @staticmethod
    def forward(ctx, input,grid,grad_output):
        ctx.save_for_backward(input,grid,grad_output)
        return GridSamplerMine.backward(input,grid,grad_output,0,1)
    @staticmethod
    def backward(ctx, grad_output_input, grad_output_grid):
        input,grid,grad_output=ctx.saved_tensors
        # if torch.isnan(input).any() or torch.isnan(grid).any() or torch.isnan(grad_output).any() or torch.isnan(grad_output_input).any() or torch.isnan(grad_output_grid).any():
        #     print('dbackward0')
        #     assert(False)
        o0,o1,o2=GridSamplerMine.dbackward(grad_output_input,grad_output_grid,input,grid,grad_output,0,1)
        # if torch.isnan(o0).any():
        #     print('dbackward1_0')
        #     assert(False)
        # if torch.isnan(o1).any():
        #     print('dbackward1_1')
        #     assert(False)
        # if torch.isnan(o2).any():
        #     print('dbackward1_2')
        #     assert(False)
        return o0,o1,o2



# class GridSamplerMine3d(nn.Module):
#     def __init__(self):
#         super(GridSamplerMine3d, self).__init__()
#         self.balance_value = balance_value

#     def forward(self, input):
#         return Interp2xBoundary3dFunction.apply(input, self.balance_value)

# class testFunc(Function):
#     @staticmethod
#     def forward(ctx,input,test):
#         ctx.save_for_backward(input)
#         ctx.L=test
#         with torch.no_grad():
#             return test(input)
#     @staticmethod
#     def backward(ctx,grad):
#         input=ctx.saved_tensors[0]
#         input.requires_grad=True
#         out=ctx.L(input)
#         print(out)
#         return torch.autograd.grad(out,input)[0]


            
