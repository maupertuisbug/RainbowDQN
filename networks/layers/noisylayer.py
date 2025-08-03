import torch 
import math 


class NoisyLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma_init=0.5):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        self.reset_parameters()
        self.training = True

    def set_training(self, mode):
        self.training = mode

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_sigma.data.fill_(self.sigma_init) 

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_mu is not None else None
        return torch.nn.functional.linear(input, weight, bias)

