import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, sigma_init=0.4):
#         super(NoisyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.sigma_init = sigma_init

#         # Initialize \mu using uniform distribution
#         self.mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(
#             -1 / math.sqrt(in_features), 1 / math.sqrt(in_features)
#         ))

#         # Initialize \sigma
#         self.sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init / math.sqrt(in_features)))

#         # Buffers to hold noise values
#         self.register_buffer("noise_weight", torch.zeros(out_features, in_features))
#         self.register_buffer("noise_bias", torch.zeros(out_features))

#         # Initialize bias for completeness
#         self.mu_bias = nn.Parameter(torch.empty(out_features).uniform_(
#             -1 / math.sqrt(in_features), 1 / math.sqrt(in_features)
#         ))
#         self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init / math.sqrt(in_features)))

#     def forward(self, x):
#         # Sample noise for forward pass
#         self.noise_weight = torch.normal(mean=0, std=1, size=self.noise_weight.shape).to(x.device)
#         self.noise_bias = torch.normal(mean=0, std=1, size=self.noise_bias.shape).to(x.device)

#         # Compute noisy weights and biases
#         weight = self.mu + self.sigma * self.noise_weight
#         bias = self.mu_bias + self.sigma_bias * self.noise_bias

#         return F.linear(x, weight, bias)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, delta_init):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_delta = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_delta = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # Initialization
        self.delta_init = delta_init
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mu using uniform distribution
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Initialize delta
        nn.init.constant_(self.weight_delta, self.delta_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_delta, self.delta_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, input):
        if self.training:
            # Generate factorized Gaussian noise
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon = torch.ger(epsilon_out, epsilon_in)  # Outer product
            self.bias_epsilon = epsilon_out

            # Add noise to weights and biases
            weight = self.weight_mu + self.weight_delta * self.weight_epsilon
            bias = self.bias_mu + self.bias_delta * self.bias_epsilon
        else:
            # No noise during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(input, weight, bias)