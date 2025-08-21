import torch
from networks.layers.noisylayer import NoisyLayer
import numpy





class Network(torch.nn.Module):
    def __init__(self, input_dim, action_dim, device, **kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = action_dim

        self.conv = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  
                                torch.nn.ReLU())
        
        flatten_size = self._get_output_size(( 4, 84, 84))
        
        noisyLayer = kwargs.get("noisyLayer", 0)
        if noisyLayer == 0:
            self.valuefunction = torch.nn.Sequential(
                                torch.nn.Linear(flatten_size, 512),
                                torch.nn.ReLU(),
                                torch.nn.Linear(512, self.output_dim)
            )
        else :
            self.valuefunction = torch.nn.Sequential(
                                NoisyLayer(flatten_size, 512),
                                torch.nn.ReLU(),
                                NoisyLayer(512, self.output_dim)
            )

    def forward(self, x, a):

        if a.dtype == numpy.dtype('int64') :
            a = torch.tensor([[a]], device = self.device)
        if x.ndim == 3:
            x_out = self.conv(x.unsqueeze(0)/255.0)
        else :
            x_out = self.conv(x/255.0)
        
        x_out = x_out.view(x_out.size(0), -1)
        value = self.valuefunction(x_out)
        q_value = torch.gather(value, dim=1, index=a)
        return q_value

    def optimal_action(self, x):
        with torch.no_grad():
            if x.ndim == 3:
                x_out = self.conv(x.unsqueeze(0)/255.0)
            else :
                x_out = self.conv(x/255.0)
            
            x_out = x_out.view(x_out.size(0), -1)
            value = self.valuefunction(x_out)
            a = torch.argmax(value,dim=1).unsqueeze(1).to(self.device)
            return a

    def setEvaluationMode(self, eval):
        for module in self.modules():
            if isinstance(module, NoisyLayer):
                module.set_training(not eval)

    def _get_output_size(self, shape):
        x_out = self.conv(torch.zeros(1, *shape))
        x_out = torch.tensor(x_out.shape[1:])
        return int(torch.prod(x_out))

