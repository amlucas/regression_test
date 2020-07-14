#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import pathlib
import torch
import torch.nn as nn
from torchdiffeq import odeint

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, *,
                 times, hl_dim: int=10, hl_num: int=3,
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

        num_ode_params = 3 # I0, beta, gamma

        self.params_nets = []
        for _ in range(num_ode_params):
            self.params_nets.append(self.get_network(input_size, output_size, hl_num, hl_dim))

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

    def forwardSir(self, inputs, times):
        gamma_net, beta_net, I0_net = self.params_nets
        # TODO: IS THEIR EFFECT SMOOTH ? IF NOT REPLACE TANH WITH RELU
        # TODO: TAKE INTO ACCOUNT SCALING IN A SMART WAY
        # TODO: INNER LOOP IN EPOCHS, DATA FEED IN BATCHES (MINI-BATCHES)

        sp = torch.nn.Softplus()
        gamma = sp(gamma_net(inputs)).reshape((-1))
        beta = sp(beta_net(inputs)).reshape((-1))
        I0 = sp(I0_net(inputs)).reshape((-1))

        N = torch.full_like(I0, self.N)
        S0 = N - I0
        R0 = torch.zeros_like(S0)
        x0 = torch.stack([S0, I0, R0], dim=1)
        x = self.integrate(x0, times, beta, gamma, N)
        output = x[:,:,1:2] # I
        return output.reshape((-1,self.output_size))

    def integrate(self, x0, times, beta, gamma, N):
        def rhs(t, x):
            S = x[:,0]
            I = x[:,1]
            R = x[:,2]
            y = torch.zeros_like(x)
            tmp = -beta * S * I / N
            y[:,0] = -beta * S * I / N
            y[:,1] =  beta * S * I / N - gamma * I
            y[:,2] =  gamma * I
            return y

        pred_y = odeint(rhs, x0, times)
        return pred_y


    def forward(self, data):
        return self.forwardSir(data, self.times)

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
    def __init__(self, folder: str):
        PATH = pathlib.Path(folder)
        data = torch.load(PATH / "data.pt")

        self.nregions = data['nregions']
        self.in_shift = data['in_shift']
        self.in_scale = data['in_scale']
        self.out_shift = data['out_shift']
        self.out_scale = data['out_scale']

        self.out_channels = data["output_names"]

        input_size = len(data["input_names"])
        output_size = self.nregions * len(self.out_channels)

        self.model = Network(input_size, output_size, folder="./surrogate_data")


        self.model.load_model()

    def get_out_channels(self):
        return self.out_channels

    # def evaluate(self, x):
    #     x = np.array(x)
    #     x = (x - self.in_shift[np.newaxis,:]) / self.in_scale[np.newaxis,:]

    #     with torch.no_grad():
    #         x = torch.Tensor(x)
    #         y = self.model.forward(x).numpy()

    #     y = (y * self.out_scale[np.newaxis,:]) + self.out_shift[np.newaxis,:]
    #     return y
