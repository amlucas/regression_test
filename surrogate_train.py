#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import argparse
import numpy as np
import pathlib
import torch

from surrogate_model import create_model

def get_transform(data, *, axis=1):
    # _max = np.max(data, axis=axis)
    # _min = np.min(data, axis=axis)
    # shift = (_min + _max) / 2
    # scale = _max - _min
    shift = np.mean(data, axis=axis)
    scale = np.std(data, axis=axis)
    return shift, scale

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

def train_surrogate(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inout_data', type=str, help=".pt file that contains th data to train on.")
    parser.add_argument('--output-dir', type=str, default="surrogate_data", help="directory that will contain all surrogates")
    parser.add_argument('--train-proportion', type=float, default=0.8, help="proportion of data used for training.")
    parser.add_argument('--max-epoch', type=int, default=20000, help="Maximum number of training epochs.")
    args = parser.parse_args(argv)

    OUT_PATH = pathlib.Path(args.output_dir)
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    inout_data = torch.load(args.inout_data)
    inputs = inout_data["inputs"]
    outputs = inout_data["outputs"]

    input_dims, ndata = inputs.shape
    output_dims, ndata_ = outputs.shape
    assert ndata == ndata_, f"input data has {ndata} samples while output has {ndata_} samples"

    # Scale the data.

    in_shift, in_scale = get_transform(inputs, axis=1)
    out_shift, out_scale = get_transform(outputs, axis=1)

    inputs = (inputs - in_shift[:,np.newaxis]) / in_scale[:,np.newaxis]
    outputs = (outputs - out_shift[:,np.newaxis]) / out_scale[:,np.newaxis]

    # Shuffle the data ans split in training / test set.

    order = np.arange(ndata)
    np.random.shuffle(order)
    inputs = inputs[:,order]
    outputs = outputs[:,order]

    x = torch.from_numpy(inputs.astype(np.float32).transpose())
    y = torch.from_numpy(outputs.astype(np.float32).transpose())

    ntrain = int(args.train_proportion * len(x))

    xtrain = x[:ntrain,:]
    ytrain = y[:ntrain,:]
    xvalid = x[ntrain:,:]
    yvalid = y[ntrain:,:]

    model = create_model(input_size = input_dims,
                         output_size = output_dims)
    model.apply(init_weights)

    # Optimize.
    best_valid_loss = 1e20
    num_worse_valid_losses = 0
    patience = 5
    max_number_of_rounds = 10
    number_of_rounds = 0
    learning_rate = 0.1
    lr_reduction_factor = 0.5

    criterion = torch.nn.MSELoss()

    for epoch in range(args.max_epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Forward pass: Compute predicted y by passing x to the model.
        y_pred = model(xtrain)
        y_pred_valid = model(xvalid)

        # Compute and print loss.
        train_loss = criterion(y_pred, ytrain)

        with torch.no_grad():
            valid_loss = criterion(y_pred_valid, yvalid)

        if epoch % 100 == 0:
            print(f"epoch {epoch}: training loss {train_loss.item()}, valid loss {valid_loss.item()}")

        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            num_worse_valid_losses = 0
        else:
            num_worse_valid_losses += 1

        if num_worse_valid_losses > patience:
            num_worse_valid_losses = 0
            number_of_rounds += 1
            learning_rate *= lr_reduction_factor
            print(f"round {number_of_rounds}: reduced lr from {learning_rate/lr_reduction_factor} to {learning_rate}")

        if number_of_rounds >= max_number_of_rounds:
            break

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    torch.save(model.state_dict(), OUT_PATH / "model.pt")
    torch.save({"xtrain": xtrain,
                "ytrain": ytrain,
                "xvalid": xvalid,
                "yvalid": yvalid,
                "in_scale": in_scale,
                "in_shift": in_shift,
                "out_scale": out_scale,
                "out_shift": out_shift,
                "nregions": inout_data['nregions'],
                "input_names": inout_data['input_varnames'],
                "output_names": inout_data['output_varnames']},
               OUT_PATH / "data.pt")


def main(argv):
    train_surrogate(argv)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
