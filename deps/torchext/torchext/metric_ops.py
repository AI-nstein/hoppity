import torch
import extlib
try:
    import extlib_cuda
except:
    print('not loading cuda metric ops')
from torch.autograd import Function
from torch.nn import Module

class NNDistanceFunc(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert len(xyz1.shape) == 3, "NnDistance requires xyz1 be of shape (batch,#points,3)"
        assert xyz1.shape[2] == 3, "NnDistance only accepts 3d point set xyz1"
        assert len(xyz2.shape) == 3, "NnDistance requires xyz2 be of shape (batch,#points,3)"
        assert xyz2.shape[2] == 3, "NnDistance only accepts 3d point set xyz2"
        assert xyz1.shape[0] == xyz2.shape[0], "NnDistance expects xyz1 and xyz2 have same batch size"

        if not xyz1.is_cuda:
            d1, i1, d2, i2 = extlib.nn_distance_forward(xyz1, xyz2)
        else:
            d1, i1, d2, i2 = extlib_cuda.nn_distance_forward_cuda(xyz1, xyz2)

        ctx.save_for_backward(xyz1, xyz2, i1, i2)
        return d1, i1, d2, i2

    @staticmethod
    def backward(ctx, grad_d1, grad_i1, grad_d2, grad_i2):
        xyz1, xyz2, i1, i2 = ctx.saved_variables
        if not grad_d1.is_cuda:
            grad_xyz1, grad_xyz2 = extlib.nn_distance_backward(xyz1, xyz2, grad_d1, i1, grad_d2, i2)
        else:
            grad_xyz1, grad_xyz2 = extlib_cuda.nn_distance_backward_cuda(xyz1, xyz2, grad_d1, i1, grad_d2, i2)
        
        return grad_xyz1, grad_xyz2


class NNDistance(Module):
    def forward(self, xyz1, xyz2):
        return NNDistanceFunc.apply(xyz1, xyz2)

nn_distance = NNDistance()


class ApproxMatchFunc(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert len(xyz1.shape) == 3, "ApproxMatch requires xyz1 be of shape (batch,#points,3)"
        assert xyz1.shape[2] == 3, "ApproxMatch only accepts 3d point set xyz1"
        assert len(xyz2.shape) == 3, "ApproxMatch requires xyz2 be of shape (batch,#points,3)"
        assert xyz2.shape[2] == 3, "ApproxMatch only accepts 3d point set xyz2"
        assert xyz1.shape[0] == xyz2.shape[0], "ApproxMatch expects xyz1 and xyz2 have same batch size"

        if not xyz1.is_cuda:
            return extlib.approxmatch_forward(xyz1, xyz2)
        else:
            return extlib_cuda.approxmatch_forward_cuda(xyz1, xyz2)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


class ApproxMatch(Module):
    def forward(self, xyz1, xyz2):
        return ApproxMatchFunc.apply(xyz1, xyz2)


approx_match = ApproxMatch()


class MatchCostFunc(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, match):
        assert len(xyz1.shape) == 3, "MatchCost requires xyz1 be of shape (batch,#points,3)"
        assert xyz1.shape[2] == 3, "MatchCost only accepts 3d point set xyz1"
        assert len(xyz2.shape) == 3, "MatchCost requires xyz2 be of shape (batch,#points,3)"
        assert xyz2.shape[2] == 3, "MatchCost only accepts 3d point set xyz2"
        assert xyz1.shape[0] == xyz2.shape[0], "MatchCost expects xyz1 and xyz2 have same batch size"

        assert len(match.shape) == 3 and match.shape[0] == xyz1.shape[0] and match.shape[1] == xyz2.shape[1] and match.shape[2] == xyz1.shape[1], "MatchCost expects (batch_size,#query,#dataset) match shape" 
        ctx.save_for_backward(xyz1, xyz2, match)

        if not xyz1.is_cuda:
            return extlib.matchcost_forward(xyz1, xyz2, match)
        else:
            return extlib_cuda.matchcost_forward_cuda(xyz1, xyz2, match)

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_variables
        
        if not grad_cost.is_cuda:
            grad_xyz1, grad_xyz2 = extlib.matchcost_backward(xyz1, xyz2, match)
        else:
            grad_xyz1, grad_xyz2 = extlib_cuda.matchcost_backward_cuda(xyz1, xyz2, match)

        grad_xyz1 = grad_xyz1 * grad_cost[:, None, None]
        grad_xyz2 = grad_xyz2 * grad_cost[:, None, None]

        return grad_xyz1, grad_xyz2, None


class MatchCost(Module):
    def forward(self, xyz1, xyz2, match):
        return MatchCostFunc.apply(xyz1, xyz2, match)

match_cost = MatchCost()
