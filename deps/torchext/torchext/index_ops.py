import torch
from torch.autograd import Function
from torch.nn import Module


class ReplaceRowsFunc(Function):
    @staticmethod
    def forward(ctx, mat_orig, indices, mat_new):
        ctx.row_indices = indices
        mat_dup = mat_orig.detach().clone()
        mat_dup[indices] = mat_new
        return mat_dup

    @staticmethod
    def backward(ctx, grad_out):
        indices = ctx.row_indices
        
        grad_dup = grad_out.detach().clone()        
        grad_new = grad_dup[indices]
        grad_dup[indices] = 0

        return grad_dup, None, grad_new


class ReplaceRows(Module):
    def forward(self, mat_orig, indices, mat_new):
        return ReplaceRowsFunc.apply(mat_orig, indices, mat_new)

replace_rows = ReplaceRows()
