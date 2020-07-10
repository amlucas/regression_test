#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import pathlib
import torch

def create_model(input_size: int,
                 output_size: int,
                 hl_dim = 32,
                 hl_num = 3):
    hidden = []
    for i in range(hl_num):
        hidden.append(torch.nn.Linear(hl_dim, hl_dim))
        hidden.append(torch.nn.Tanh())

    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hl_dim),
        torch.nn.Tanh(),
        *hidden,
        torch.nn.Linear(hl_dim, output_size))


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

        self.model = create_model(input_size = len(data["input_names"]),
                                  output_size = self.nregions * len(self.out_channels))
        self.model.load_state_dict(torch.load(PATH / "model.pt"))

    def get_out_channels(self):
        return self.out_channels

    def evaluate(self, x):
        x = np.array(x)
        x = (x - self.in_shift[np.newaxis,:]) / self.in_scale[np.newaxis,:]

        with torch.no_grad():
            x = torch.Tensor(x)
            y = self.model(x).numpy()

        y = (y * self.out_scale[np.newaxis,:]) + self.out_shift[np.newaxis,:]
        return y
