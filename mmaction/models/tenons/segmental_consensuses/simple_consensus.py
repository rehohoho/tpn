import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES


class _SimpleConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, consensus_type='avg', dim=1):
        shape = x.size()
        
        # save_for_backward only for tensors
        # stores as SavedVariables for sanity checks
        ctx.shape = shape
        ctx.dim = dim
        ctx.consensus_type = consensus_type

        if consensus_type == 'avg':
            return x.mean(dim=dim, keepdim=True)
        else:
            return None
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            return grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        else:
            return None


@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus.apply(input, self.consensus_type, self.dim)
