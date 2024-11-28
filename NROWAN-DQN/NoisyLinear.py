import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Initialize \mu using uniform distribution
        self.mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(
            -1 / math.sqrt(in_features), 1 / math.sqrt(in_features)
        ))

        # Initialize \sigma
        self.sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init / math.sqrt(in_features)))

        # Buffers to hold noise values
        self.register_buffer("noise_weight", torch.zeros(out_features, in_features))
        self.register_buffer("noise_bias", torch.zeros(out_features))

        # Initialize bias for completeness
        self.mu_bias = nn.Parameter(torch.empty(out_features).uniform_(
            -1 / math.sqrt(in_features), 1 / math.sqrt(in_features)
        ))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init / math.sqrt(in_features)))

    def forward(self, x):
        # Sample noise for forward pass
        self.noise_weight = torch.normal(mean=0, std=1, size=self.noise_weight.shape).to(x.device)
        self.noise_bias = torch.normal(mean=0, std=1, size=self.noise_bias.shape).to(x.device)

        # Compute noisy weights and biases
        weight = self.mu + self.sigma * self.noise_weight
        bias = self.mu_bias + self.sigma_bias * self.noise_bias

        return F.linear(x, weight, bias)
