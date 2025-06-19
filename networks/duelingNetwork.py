import torch
from networks.layers.noisylayer import NoisyLayer
import numpy





class DuelingNetwork(torch.nn.Module):
    def __init__(self, input_dim, action_dim, noisyLayer, device):
        super().__init__()
        self.conv = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),  
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  
                                torch.nn.ReLU())
        
        flatten_size = self.get_output_size((1, 84, 84))
        self.device = device
        
        if noisyLayer == 0:
            self.valuefunction = torch.nn.Sequential(
                                torch.nn.Linear(flatten_size, 128),
                                torch.nn.ReLU(),
                                torch.nn.Linear(128,1)
            )
            self.advantagefunction = torch.nn.Sequential(
                                torch.nn.Linear(flatten_size, 128),
                                torch.nn.ReLU(),
                                torch.nn.Linear(128, action_dim)
            )
        else :
            self.valuefunction = torch.nn.Sequential(
                                NoisyLayer(flatten_size, 128),
                                torch.nn.ReLU(),
                                NoisyLayer(128,1)
            )
            self.advantagefunction = torch.nn.Sequential(
                                NoisyLayer( flatten_size, 128),
                                torch.nn.ReLU(),
                                NoisyLayer(128, action_dim)
            )

    def forward(self, x, a):

        if a.dtype == numpy.dtype('int64') :
            a = torch.tensor([[a]], device = self.device)
        if x.ndim == 3:
            x_out = self.conv(x.unsqueeze(0)/255.0)
        else :
            x_out = self.conv(x/255.0)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = torch.nn.functional.relu(x_out)
        value = self.valuefunction(x_out)
        advantage = self.advantagefunction(x_out)
        advantage_mean = torch.mean(advantage, dim=-1)
        advantage_value = torch.gather(advantage, dim=1, index=a)
        advantage_value = advantage_value - advantage_mean
        q_value = value + advantage_value
        return q_value

    def optimal_action(self, x):
        with torch.no_grad():
            if x.ndim == 3:
                x_out = self.conv(x.unsqueeze(0)/255.0)
            else :
                x_out = self.conv(x/255.0)
            x_out = x_out.view(x_out.size(0), -1)
            x_out = torch.nn.functional.relu(x_out)
            value = self.valuefunction(x_out)
            advantage = self.advantagefunction(x_out)
            advantage_mean = torch.mean(advantage, dim=-1).unsqueeze(1)
            advantage_values = advantage - advantage_mean
            a = torch.argmax(advantage_values,dim=1).unsqueeze(1).to(self.device)
            return a

    def get_output_size(self, shape):
        x_out = self.conv(torch.zeros(1, *shape))
        x_out = torch.tensor(x_out.shape[1:])
        return int(torch.prod(x_out))



    
