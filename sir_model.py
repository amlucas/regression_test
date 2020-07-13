#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import pathlib
import torch
import torch.nn as nn

class Network:
    def __init__(self, input_size: int, output_size: int, hl_dim = 32, hl_num = 3, folder="./surrogate_data"):

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



        self.input_size = input_size
        self.output_size = output_size
        self.hl_dim = hl_dim
        self.hl_num = hl_num
        self.path = pathlib.Path(folder)

        # gamma = 0.2
        # beta = 2 * gamma
        # N = 1e6
        # I0 = N * 0.01
        # tmax = 100
        # t, I = sir(beta=beta, gamma=gamma, I0=I0, N=N, tmax=tmax)

        hidden_dim = 10
        hidden_layers = 5
        input_size = 4
        output_size = 1
        gamma_network = self.get_network(input_size, output_size, hidden_layers, hidden_dim)
        beta_network = self.get_network(input_size, output_size, hidden_layers, hidden_dim)
        I0_network = self.get_network(input_size, output_size, hidden_layers, hidden_dim)
        self.params_layers = []
        self.params_layers.append(gamma_network)
        self.params_layers.append(beta_network)
        self.params_layers.append(I0_network)
        self.initialize_weights()

        # self.model = self.create_model(input_size, output_size, hl_dim, hl_num)

    def get_network(self, input_size: int, output_size: int, hidden_layers = 3, hidden_dim = 10):
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
        for network in self.params_layers:
            for layer in network:
                params += layer.parameters()
        return params

    def forwardSir(self, input, time):
        print("# forwardSir()")
        print(input.size())
        print(time.size())
        # N = np.array([10000])
        # N = self.torch_dtype(N)
        gamma_network, beta_network, I0_network = self.params_layers
        # TODO: ADD POSITIVE ACTIVATIONS (SOFTPLUS, ETC.)
        # TODO: IS THEIR EFFECT SMOOTH ? IF NOT REPLACE TANH WITH RELU
        # TODO: TAKE INTO ACCOUNT SCALING IN A SMART WAY 
        # TODO: INNER LOOP IN EPOCHS, DATA FEED IN BATCHES (MINI-BATCHES)
        gamma = gamma_network(input)
        beta = beta_network(input)
        I0 = I0_network(input)

        N = 10000 * torch.ones_like(I0)
        print(N.size())
        S0 = N - I0
        R0 = torch.zeros_like(S0)
        print("DIMS:")
        print(S0.size())
        print(I0.size())
        print(R0.size())
        x0 = torch.cat([S0, I0, R0], dim=1)
        t0 = torch.zeros_like(S0)
        tmax = time
        x = self.integrate(x0, t0, tmax, beta, gamma, N)
        # Keep only the final I
        output = x[:, 1:2]
        return output


    def rhs(self, x, t, beta, gamma, N):
        S, I, R = x
        rhs_ = torch.stack([
            -beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I
        ]) 
        return rhs_

    def integrate(self, x0, t0, tmax, beta, gamma, N):
        K, T = tmax.size()
        dt = 1.0
        x_all = []
        for k in range(K):
            T_max = tmax[k]
            T_max_np = T_max.cpu().detach().numpy()[0]
            num_steps = int(T_max_np/dt)

            x_k = x0[k]
            beta_k = beta[k]
            gamma_k = gamma[k]
            N_k = N[k]

            for n in range(num_steps):
                rhs_ = self.rhs(x_k, n * dt, beta_k, gamma_k, N_k)
                rhs_ = rhs_[:,0]
                # print(x_k.size())
                # print(rhs_.size())
                # print(ark)
                x_k += rhs_ * dt
                # print(x_k.size())
            x_all.append(x_k.clone())
        x_all = torch.stack(x_all)
        return x_all


    def forward(self, data):
        time = data[:,-1:]
        state = data[:,:-1]

        return self.forwardSir(state, time)


    def initialize_weights(self):
        print("Initializing parameters of the network.\n")
        for network in self.params_layers:
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



