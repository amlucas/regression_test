#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import pathlib
import torch
import torch.nn as nn
from torchdiffeq import odeint

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, *,
                 times, hl_dim: int=32, hl_num: int=3,
                 N=8.6e6, folder="./surrogate_data"):
        super().__init__()

        self.gpu = False
        # SETTING DEFAULT DATATYPE:
        if self.gpu:
            # self.torch_dtype = torch.cuda.DoubleTensor
            self.torch_dtype = torch.cuda.FloatTensor
            # torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.torch_dtype = torch.FloatTensor
            # self.torch_dtype = torch.DoubleTensor
            # torch.set_default_tensor_type(torch.DoubleTensor)
            torch.set_default_tensor_type(torch.FloatTensor)

        self.N = N
        self.input_size = input_size
        self.output_size = output_size
        self.path = pathlib.Path(folder)
        self.times = times

        self.R_net = self.get_network(input_size, 1, hl_num, hl_dim)
        self.gamma_net = self.get_network(input_size, 1, hl_num, hl_dim)
        self.I0_net = self.get_network(input_size, 1, hl_num, hl_dim)
        self.kint_net = self.get_network(input_size, 1, hl_num, hl_dim)
        self.tint_net = self.get_network(input_size, 1, hl_num, hl_dim)

        self.params_nets = []
        self.params_nets.append(self.R_net)
        self.params_nets.append(self.gamma_net)
        self.params_nets.append(self.I0_net)
        self.params_nets.append(self.kint_net)
        self.params_nets.append(self.tint_net)

        self.initialize_weights()

    def get_network(self, input_size: int, output_size: int, hidden_layers, hidden_dim):
        hidden = nn.ModuleList()
        for i in range(hidden_layers):
            hidden.append(torch.nn.Linear(hidden_dim, hidden_dim))
            hidden.append(torch.nn.Tanh())
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.Tanh(),
            *hidden,
            torch.nn.Linear(hidden_dim, output_size))

    def save_model(self):
        torch.save(self.state_dict(), self.path / "model.pt")

    def load_model(self):
        self.load_state_dict(torch.load(self.path / "model.pt"))

    def get_parameters(self):
        params = list()
        for network in self.params_nets:
            for layer in network:
                params += layer.parameters()
        return params

    def forward_sir(self, inputs, times):
        # TODO: IS THEIR EFFECT SMOOTH ? IF NOT REPLACE TANH WITH RELU
        # TODO: TAKE INTO ACCOUNT SCALING IN A SMART WAY
        # TODO: INNER LOOP IN EPOCHS, DATA FEED IN BATCHES (MINI-BATCHES)

        R_net, gamma_net, I0_net, kint_net, tint_net = self.params_nets

        sp = torch.nn.Softplus()
        gamma = sp(gamma_net(inputs))[:,0] + 0.1
        R = sp(R_net(inputs))[:,0] + 1.5
        I0 = sp(I0_net(inputs))[:,0]
        kint = sp(kint_net(inputs))[:,0]
        tint = sp(tint_net(inputs))[:,0] + 20
        beta0 = R * gamma
        beta1 = beta0 * kint

        N = torch.full_like(I0, self.N)

        #print(beta[0].item(), gamma[0].item(), I0[0].item())

        S0 = N - I0
        R0 = torch.zeros_like(S0)
        x0 = torch.stack([S0, I0, R0], dim=1)
        x = self.integrate(x0, times, beta0, beta1, tint, gamma, N)
        S = x[:,:,0]
        dI = torch.zeros_like(S)
        dI[1:,:] = S[:-1,:] - S[1:,:]

        return dI.reshape((-1,self.output_size))

    def integrate(self, x0, times, beta0, beta1, tint, gamma, N):
        def logistic(x):
            return 1 / (1 + torch.exp(-x))

        def beta(t):
            dt = 7/4
            return logistic((t - tint) / dt) * (beta1 - beta0) + beta0

        def rhs(t, x):
            S = x[:,0]
            I = x[:,1]
            R = x[:,2]
            y = torch.zeros_like(x)
            y[:,0] = -beta(t) * S * I / N
            y[:,1] =  beta(t) * S * I / N - gamma * I
            y[:,2] =  gamma * I
            return y

        pred_y = odeint(rhs, x0, times, method='rk4')
        return pred_y


    def forward(self, data):
        return self.forward_sir(data, self.times)

    def initialize_weights(self):
        print("Initializing parameters of the network...")
        for network in self.params_nets:
            for layer in network:
                # print(module)
                for name, param in layer.named_parameters():
                    # print(name)
                    # INITIALIZING RNN, GRU CELLS
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)

                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn(["Wxi.weight", "Wxf.weight", "Wxc.weight", "Wxo.weight"], name):
                        torch.nn.init.xavier_uniform_(param.data)

                    elif self.ifAnyIn(["Wco", "Wcf", "Wci", "Whi.weight", "Whf.weight", "Whc.weight", "Who.weight"], name):
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn(["Whi.bias", "Wxi.bias", "Wxf.bias", "Whf.bias", "Wxc.bias", "Whc.weight", "Wxo.bias", "Who.bias"], name):
                        param.data.fill_(0)

                    elif 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)

                    elif 'bias' in name:
                        param.data.fill_(0)
                    else:
                        raise ValueError("NAME {:} NOT FOUND!".format(name))
                        # print("NAME {:} NOT FOUND!".format(name))
        print("Parameters initialized.")
        return 0

    def ifAnyIn(self, list_, name):
        for element in list_:
            if element in name:
                return True
        return False

class Surrogate:
    def __init__(self, times, folder: str):
        PATH = pathlib.Path(folder)
        data = torch.load(PATH / "data.pt")

        self.nregions = data['nregions']
        self.in_shift = data['in_shift']
        self.in_scale = data['in_scale']
        self.out_shift = data['out_shift']
        self.out_scale = data['out_scale']

        self.out_channels = data["output_names"]

        input_size = len(data["input_names"]) - 1
        output_size = self.nregions * 1

        self.model = Network(input_size, output_size, times = torch.from_numpy(times.astype(np.float32)), folder="./surrogate_data")
        self.model.load_model()

    def get_out_channels(self):
        return self.out_channels

    def evaluate(self, x):
        x = np.array(x)
        x = (x - self.in_shift[np.newaxis,:]) / self.in_scale[np.newaxis,:]
        x = torch.from_numpy(x.astype(np.float32))

        with torch.no_grad():
            x = torch.Tensor(x)
            y = self.model.forward(x).numpy()

        #y = (y * self.out_scale[np.newaxis,:]) + self.out_shift[np.newaxis,:]
        return y
