import torch
import extlib
try:
    import extlib_cuda
except:
    print('not loading cuda jagged ops')
from torch.autograd import Function
from torch.nn import Module
import numpy as np


#----------------------
#   jagged_argmax
#----------------------
class JaggedArgmaxFunc(Function):
    @staticmethod
    def forward(ctx, values, prefix_sum):
        assert len(prefix_sum.size()) == 1
        if not values.is_cuda:
            return extlib.jagged_argmax_forward(values, prefix_sum)
        else:
            return extlib_cuda.jagged_argmax_forward_cuda(values, prefix_sum)

    @staticmethod
    def backward(ctx, grad_output):
        assert False


class JaggedArgmax(Module):
    def forward(self, values, prefix_sum):
        return JaggedArgmaxFunc.apply(values, prefix_sum)

jagged_argmax = JaggedArgmax()

#----------------------
#   jagged2padded
#----------------------
class Jagged2PaddedFunc(Function):
    @staticmethod
    def forward(ctx, values, prefix_sum, pad_val, pad_size=None):
        assert len(prefix_sum.size()) == 1
        if pad_size is None:  # pad with max len
            seglens = torch.cat([prefix_sum[0:1], prefix_sum[1:] - prefix_sum[:-1]])
            pad_size = torch.max(seglens).item()
        padded = values.new(prefix_sum.shape[0], pad_size)
        if not padded.is_contiguous():
            padded.contiguous()
        padded.fill_(pad_val)
        if not values.is_cuda:
            extlib.jagged2padded_forward(values, prefix_sum, padded)
        else:
            extlib_cuda.jagged2padded_forward_cuda(values, prefix_sum, padded)
        return padded

    @staticmethod
    def backward(ctx, grad_output):
        assert False

class Jagged2Padded(Module):
    def forward(self, values, prefix_sum, pad_val, pad_size=None):
        return Jagged2PaddedFunc.apply(values, prefix_sum, pad_val, pad_size)

jagged2padded = Jagged2Padded()

#----------------------
#   jagged_topk
#----------------------
class JaggedTopkFunc(Function):
    @staticmethod
    def forward(ctx, values, prefix_sum, k, largest=True):
        assert len(prefix_sum.size()) == 1        
        pad_val = np.finfo(np.float32).min if largest else np.finfo(np.float32).max
        padded_val = jagged2padded(values, prefix_sum, pad_val)
        if k > padded_val.shape[1]:
            k = padded_val.shape[1]
        return torch.topk(padded_val, k, dim=-1, largest=largest)

    @staticmethod
    def backward(ctx, grad_output):
        assert False

class JaggedTopk(Module):
    def forward(self, values, prefix_sum, k, largest=True):
        return JaggedTopkFunc.apply(values, prefix_sum, k, largest)

jagged_topk = JaggedTopk()

#----------------------
#   jagged_log_softmax
#----------------------
class JaggedLogSoftmaxFunc(Function):
    @staticmethod
    def forward(ctx, logits, prefix_sum):
        assert len(prefix_sum.size()) == 1        
        if not logits.is_cuda:
            output = extlib.jagged_log_softmax_forward(logits, prefix_sum)
        else:
            output = extlib_cuda.jagged_log_softmax_forward_cuda(logits, prefix_sum)

        ctx.save_for_backward(prefix_sum, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        prefix_sum, output = ctx.saved_variables
        if not grad_output.is_cuda:
            grad_input = extlib.jagged_log_softmax_backward(output.data, grad_output, prefix_sum.data)
        else:
            grad_input = extlib_cuda.jagged_log_softmax_backward_cuda(output.data, grad_output, prefix_sum.data)        
        return grad_input, None


class JaggedLogSoftmax(Module):
    def forward(self, logits, prefix_sum):
        return JaggedLogSoftmaxFunc.apply(logits, prefix_sum)

jagged_log_softmax = JaggedLogSoftmax()


#----------------------
#   jagged_append
#----------------------
class JaggedAppendFunc(Function):
    @staticmethod
    def forward(ctx, values, prefix_sum, suffix_mat):
        assert len(prefix_sum.size()) == 1
        assert len(suffix_mat.size()) == 2
        assert prefix_sum.shape[0] == suffix_mat.shape[0]

        output = values.new(values.shape[0] + suffix_mat.shape[1] * suffix_mat.shape[0])
        if not values.is_cuda:
            extlib.jagged_append_forward(values, prefix_sum, suffix_mat, output)
        else:
            extlib_cuda.jagged_append_forward_cuda(values, prefix_sum, suffix_mat, output)

        ctx.suffix_len = suffix_mat.shape[-1]
        ctx.save_for_backward(prefix_sum)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        prefix_sum = ctx.saved_variables[0]
        val_num_eles = prefix_sum[-1].item()
        grad_val = grad_output.new(val_num_eles)
        grad_suffix = grad_output.new(prefix_sum.shape[0], ctx.suffix_len)

        if not grad_output.is_cuda:
            extlib.jagged_append_backward(grad_output, prefix_sum.data, grad_val, grad_suffix)
        else:
            extlib_cuda.jagged_append_backward_cuda(grad_output, prefix_sum.data, grad_val, grad_suffix)
        return grad_val, None, grad_suffix


class JaggedAppend(Module):
    def forward(self, values, prefix_sum, suffix_mat):
        return JaggedAppendFunc.apply(values, prefix_sum, suffix_mat)

jagged_append = JaggedAppend()
