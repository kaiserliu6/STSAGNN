import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dim=-2, num_layer=1):
        super().__init__()
        self.num_layer = num_layer
        # self.cell = nn.ModuleList([GRUCell(input_size, hidden_size)])
        self.cell = nn.ModuleList([nn.GRUCell(input_size, hidden_size)])
        for i in range(1, num_layer):
            self.cell.append(GRUCell(hidden_size, hidden_size))
        self.d = hidden_size
        self.dim = dim

    def forward(self, x, h=None):
        # you can input whatever size you want, just make sure last dim is for dim, and provide the dim of sequence
        x = x.transpose(-2, self.dim)
        shape = x.shape
        current_input = x.flatten(0, -3) #b, t, d
        if h is not None:
            init_state = [h.flatten(0, -2)]
        else:
            init_state = [torch.zeros(current_input.shape[0], self.d, device=x.device) for i in range(self.num_layer)]
        for i in range(self.num_layer):
            state = init_state[i]
            hidden_state = []
            for j in range(current_input.shape[1]):
                state = self.cell[i](current_input[:, j], state)
                hidden_state.append(state)
            current_input = torch.stack(hidden_state, 1)
        x = current_input.unflatten(0, shape[:-2]).transpose(-2, self.dim)
        return x

class GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dim=-2, num_layer=1, add_extern_factor=False):
        super().__init__()
        self.num_layer = num_layer
        self.d = hidden_size
        self.d_out = output_size
        self.dim = dim
        # self.emb = nn.Linear(output_size, hidden_size)
        self.proj = nn.Linear(hidden_size, output_size)
        # n = 66
        # self.emb = OD_embedder(hidden_size, output_size, n)
        # self.proj = OD_predictor(hidden_size, output_size, n)
        self.add_extern_factor = add_extern_factor
        if add_extern_factor:
            # self.cell = nn.ModuleList([GRUCell(input_size + output_size, hidden_size)])
            # self.cell = nn.ModuleList([GRUCell(input_size + hidden_size, hidden_size)])
            self.cell = nn.ModuleList([GRUCell(hidden_size, hidden_size)])
            for i in range(1, num_layer):
                self.cell.append(GRUCell(hidden_size, hidden_size))
        else:
            # self.cell = nn.ModuleList([GRUCell(output_size, hidden_size)])
            self.cell = nn.ModuleList([GRUCell(hidden_size, hidden_size)])
            for i in range(1, num_layer):
                self.cell.append(GRUCell(hidden_size, hidden_size))

    def forward(self, x, h):
        # you can input whatever size you want, just make sure last dim is for dim
        """
        :param x: *, input_size if add_extern_factor=True else int:horizon
        :param h: num_layers, *, hidden_size
        :return: *, horizon, output_size
        """
        if isinstance(h, list):
            shape = h[0].shape
            h = [i.flatten(0, -2) for i in h] #b,d
        else:
            shape = h.shape
            h = [h.flatten(0, -2)]
        output = []
        b = h[0].shape[0]
        if self.add_extern_factor:
            x = x.transpose(-2, self.dim)
            # shape = x.shape
            extern_list = x.flatten(0, -3) #b,t,d
            win_out = extern_list.shape[1]
            current_output = torch.zeros(b, self.d_out, device=h[0].device)  # b,do
            for i in range(win_out):
                # state = torch.cat((current_output, extern_list[:, i]), -1)
                # state = torch.cat((self.emb(current_output), extern_list[:, i]), -1)
                # state = torch.cat((h[-1], extern_list[:, i]), -1)
                state = extern_list[:, i]
                for j in range(self.num_layer):
                    state = self.cell[j](state, h[j])
                    h[j] = state
                # output.append(self.proj(state))
                current_output = self.proj(state)
                output.append(current_output)
        else:
            win_out = x
            current_output = torch.zeros(b, self.d_out, device=h[0].device)  # b,do
            for i in range(win_out):
                # state = current_output
                # state = self.emb(current_output)
                state = h[-1]
                for j in range(self.num_layer):
                    state = self.cell[j](state, h[j])
                    h[j] = state
                current_output = self.proj(state)
                output.append(current_output)
        output = torch.stack(output, -2).unflatten(0, shape[:-1]).transpose(-2, self.dim)
        return output

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.output = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, h):
        #input ..., D
        xh = torch.cat((input, h), dim=-1)
        gates = self.gate(xh)
        r, z = gates.chunk(2, -1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        h_ = torch.cat((input, r * h), dim=-1)
        h_ = torch.tanh(self.output(h_))
        h = z * h + (1 - z) * h_
        return h

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1):
        super().__init__()

        self.num_layer = num_layer
        self.cell = nn.ModuleList([LSTMCell(input_size, hidden_size)])
        for i in range(1, num_layer):
            self.cell.append(LSTMCell(hidden_size, hidden_size))
        self.d = hidden_size

    def forward(self, x):
        b, t, n, d = x.size()
        h = [torch.zeros(b, n, self.d).to(x.device) for i in range(self.num_layer)]
        c = [torch.zeros(b, n, self.d).to(x.device) for i in range(self.num_layer)]
        for i in range(t):
            h[0], c[0] = self.cell[0](x[:, i], (h[0], c[0]))
            for j in range(1, self.num_layer):
                h[j], c[j] = self.cell[j](h[j-1], (h[j], c[j]))
        return h[-1]

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, input, hx):
        #input B, N, D
        h, c = hx
        xh = torch.cat((input, h), dim=-1)
        gates = self.gate(xh)
        i, f, cg, o = gates.chunk(4, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        cg = torch.tanh(cg)
        o = torch.sigmoid(o)
        cy = f * c + i * cg
        hy = o * torch.tanh(cy)
        return hy, cy