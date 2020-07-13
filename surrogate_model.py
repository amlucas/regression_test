#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import pathlib
import torch

class Network:
    def __init__(self, input_size: int, output_size: int, hl_dim = 32, hl_num = 3, folder="./surrogate_data"):
        self.input_size = input_size
        self.output_size = output_size
        self.hl_dim = hl_dim
        self.hl_num = hl_num
        self.model = self.create_model(input_size, output_size, hl_dim, hl_num)
        self.initialize_weights()
        self.path = pathlib.Path(folder)

    def create_model(self, input_size: int, output_size: int, hl_dim = 32, hl_num = 3):
        hidden = []
        for i in range(hl_num):
            hidden.append(torch.nn.Linear(hl_dim, hl_dim))
            hidden.append(torch.nn.Tanh())
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hl_dim),
            torch.nn.Tanh(),
            *hidden,
            torch.nn.Linear(hl_dim, output_size))

    def save_model(self):
        torch.save(self.model.state_dict(), self.path / "model.pt")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path / "model.pt"))

    def get_parameters(self):
        params = list()
        for layer in self.model:
            params += layer.parameters()
        return params

    def forward(self, data):
        return self.model(data)


    def initialize_weights(self):
        print("Initializing parameters of the network.\n")
        for module in self.model:
            # print(module)
            for name, param in module.named_parameters():
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

    def evaluate(self, x):
        x = np.array(x)
        x = (x - self.in_shift[np.newaxis,:]) / self.in_scale[np.newaxis,:]

        with torch.no_grad():
            x = torch.Tensor(x)
            y = self.model.forward(x).numpy()

        y = (y * self.out_scale[np.newaxis,:]) + self.out_shift[np.newaxis,:]
        return y
