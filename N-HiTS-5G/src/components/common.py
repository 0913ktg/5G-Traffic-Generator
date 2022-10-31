# Cell
import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd.function import Function

# Cell
class Chomp1d(nn.Module):
    """
    Receives x input of dim [N,C,T], and trims it so that only
    'time available' information is used. Used for one dimensional
    causal convolutions.
    : param chomp_size: lenght of outsample values to skip.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

class CausalConv1d(nn.Module):
    """
    Receives x input of dim [N,C,T], computes a unidimensional
    causal convolution.

    Parameters
    ----------
    in_channels: int
    out_channels: int
    activation: str
        https://discuss.pytorch.org/t/call-activation-function-from-string
    padding: int
    kernel_size: int
    dilation: int

    Returns:
    x: tesor
        torch tensor of dim [N,C,T]
        activation(conv1d(inputs, kernel) + bias)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, activation, stride:int=1, with_weight_norm:bool=False):
        super(CausalConv1d, self).__init__()
        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'

        self.conv       = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation)
        if with_weight_norm: self.conv = weight_norm(self.conv)

        self.chomp      = Chomp1d(padding)
        self.activation = getattr(nn, activation)()
        self.causalconv = nn.Sequential(self.conv, self.chomp, self.activation)

    def forward(self, x):
        return self.causalconv(x)

# Cell
class TimeDistributed2d(nn.Module):
    """
    Receives x input of dim [N,C,T], reshapes it to [T,N,C]
    Collapses input of dim [T,N,C] to [TxN,C] and applies module to C.
    Finally reshapes it to [N,C_out,T].
    Allows handling of variable sequence lengths and minibatch sizes.
    : param module: Module to apply input to.
    """
    def __init__(self, module):
        super(TimeDistributed2d, self).__init__()
        self.module = module

    def forward(self, x):
        N, C, T = x.size()
        x = x.permute(2, 0, 1).contiguous()
        x = x.view(T * N, -1)
        x = self.module(x)
        x = x.view(T, N, -1)
        x = x.permute(1, 2, 0).contiguous()
        return x

# Cell
class TimeDistributed3d(nn.Module):
    """
    Receives x input of dim [N,L,C,T], reshapes it to [T,N,L,C]
    Collapses input of dim [T,N,L,C] to [TxNxL,C] and applies module to C.
    Finally reshapes it to [N,L,C_out,T].
    Allows handling of variable sequence lengths and minibatch sizes.
    : param module: Module to apply input to.
    """
    def __init__(self, module):
        super(TimeDistributed3d, self).__init__()
        self.module = module

    def forward(self, x):
        N, L, C, T = x.size()
        x = x.permute(3, 0, 1, 2).contiguous() #[N,L,C,T] --> [T,N,L,C]
        x = x.view(T * N * L, -1)
        x = self.module(x)
        x = x.view(T, N, L, -1)
        x = x.permute(1, 2, 3, 0).contiguous() #[T,N,L,C] --> [N,L,C,T]
        return x

# Cell
class RepeatVector(nn.Module):
    """
    Receives x input of dim [N,C], and repeats the vector
    to create tensor of shape [N, C, K]
    : repeats: int, the number of repetitions for the vector.
    """
    def __init__(self, repeats):
        super(RepeatVector, self).__init__()
        self.repeats = repeats

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, self.repeats) # <------------ Mejorar?
        return x

# Cell
class L1Regularizer(nn.Module):
    """
    Layer meant to apply elementwise L1 regularization to a dimension.
    Receives x input of dim [N,C] and returns the input [N,C].
    """
    def __init__(self, in_features, l1_lambda):
        super(L1Regularizer, self).__init__()
        self.l1_lambda = l1_lambda
        self.weight = t.nn.Parameter(t.rand((in_features), dtype=t.float),
                                     requires_grad=True)

    def forward(self, x):
        # channelwise regularization, turns on or off channels
        x = t.einsum('bp,p->bp', x, self.weight)
        return x

    def regularization(self):
        return self.l1_lambda * t.norm(self.weight, 1)