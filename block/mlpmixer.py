import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Mixer(nn.Module):
    def __init__(self, input_size, hidden_size = None, output_size = None, num_layer = 1, dim = (-1)):
        super().__init__()

        hidden_size = hidden_size or input_size
        output_size = output_size or input_size
        if isinstance(dim, int):
            dim = [dim]
            if isinstance(input_size, int):
                input_size = [input_size]
            if isinstance(hidden_size, int):
                hidden_size = [hidden_size]
            if isinstance(output_size, int):
                output_size = [output_size]
        assert len(dim) == len(input_size)
        model = []
        for j in range(num_layer - 1):
            for i in range(len(dim)):
                model.append(MLP_Mixer_Layer(input_size[i], hidden_size[i], input_size[i], dim[i]))
        for i in range(len(dim)):
            model.append(MLP_Mixer_Layer(input_size[i], hidden_size[i], output_size[i], dim[i]))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class MLP_Mixer_Layer(nn.Module):
    def __init__(self, input_size, hidden_size = None, output_size = None, dim = -1):
        super().__init__()
        output_size = output_size or input_size
        hidden_size = hidden_size or input_size
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.af = nn.GELU()
        self.norm = nn.LayerNorm(input_size)
        self.dim = dim
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(-1, self.dim)
        y = self.lin2(self.af(self.lin1(self.norm(x))))
        if x.shape == y.shape:
            y = x + y
        y = y.transpose(-1, self.dim)
        y = self.dp(y)
        return y