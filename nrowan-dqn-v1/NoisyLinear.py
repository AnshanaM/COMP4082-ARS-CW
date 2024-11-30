import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

"""
This layer defines a noisy layer, which received p-dimensions x and 
returns q-dimensions y

in_features(aka input dimensions): p
out_features(aka output dimensions): q

equation 4:
y = wx + b
y =                             w                   x +                     b
y = ((weight_mu + weight_sigma) o weight_epsilon) * x + (((bias_mu + bias_sigma) o bias_epsilon) )
where o denotes element-wise multiplications

cardinality of mu, sigma in weight = p*q
cardinality of mu, sigma in bias  = q

cardinality of epsilon in weight = p*q
cardinality of epsilon in bias = q

equation 5:
use factorised gaussian noise to generate p independent gaussian 
noises epsilon_i and q independent gaussian noises epsilon_j

weight_epsilon = sgn(epsilon_i * epsilon_j) * sqrt(mod(epsilon_i * epsilon_j))

equation 6:
bias_epsilon = sgn(epsilon_j) * sqrt(mod(epsilon_j))


where sgn() is the signum function (gets sign of a real number)

sigma = 0.4
mu_i and mu_j are sampled from independent uniform distribution U[-1/sqrt(p),1/sqrt(p)]
each sigma_i and sigma_j are initialised to sigma_init/sqrt(p)

"""


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init = 0.4):

        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # set learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # noise buffers - part of the models state not as a parameter
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # initialisation
        self.std_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # initialize mu using uniform distribution
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    # calulate factorised gaussian noise
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    # forward pass
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    # computation of D
    def sigma_loss(self):
        tensorws = torch.abs(self.weight_sigma)
        ws_mean = torch.mean(tensorws)
        tensorbs = torch.abs(self.bias_sigma)
        bs_mean = torch.mean(tensorbs)
        q = self.out_features
        p = self.in_features
        return (ws_mean*q*p + bs_mean*p)/(q*p+q)
    
    def get_sigma(self):
        tw = torch.mean(self.weight_sigma)
        tw = tw.data.cpu().numpy()
        tb = torch.mean(self.bias_sigma)
        tb = tb.data.cpu().numpy()
        return tw, tb
