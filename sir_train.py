#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import argparse
import numpy as np
import pathlib
import torch

from sir_model import Network

def get_transform(data, *, axis=1):
    # _max = np.max(data, axis=axis)
    # _min = np.min(data, axis=axis)
    # shift = (_min + _max) / 2
    # scale = _max - _min
    shift = np.mean(data, axis=axis)
    scale = np.std(data, axis=axis)
    return shift, scale


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


    # # Scale the data.

    # in_shift, in_scale = get_transform(inputs, axis=1)
    # out_shift, out_scale = get_transform(outputs, axis=1)

    # inputs = (inputs - in_shift[:,np.newaxis]) / in_scale[:,np.newaxis]
    # outputs = (outputs - out_shift[:,np.newaxis]) / out_scale[:,np.newaxis]

    # Shuffle the data and split in training / test set.

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

    print("Shape of xtrain={:}".format(xtrain.size()))
    print("Shape of ytrain={:}".format(ytrain.size()))
    print("Shape of xvalid={:}".format(xvalid.size()))
    print("Shape of yvalid={:}".format(yvalid.size()))


    model = Network(input_dims, output_dims, folder=OUT_PATH)

    xtrain = xtrain[:100]
    y_pred = model.forward(xtrain)

    y_target = ytrain[:100,0:1]
    print("Shape of y_pred={:}".format(y_pred.size()))
    print("Shape of y_target={:}".format(y_target.size()))

    print(ark)

    # Optimize.
    best_valid_loss = 1e20
    num_worse_valid_losses = 0
    patience = 5
    max_number_of_rounds = 3
    number_of_rounds = 0
    learning_rate = 1.0
    lr_reduction_factor = 0.1
    print_every = 1

    criterion = torch.nn.MSELoss()

    model.save_model()

    optimizer = torch.optim.Adam(model.get_parameters(), lr=learning_rate)

    for epoch in range(args.max_epoch):
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model.
        y_pred = model.forward(xtrain)
        y_pred_valid = model.forward(xvalid)

        # Compute and print loss.
        train_loss = criterion(y_pred, ytrain)

        with torch.no_grad():
            valid_loss = criterion(y_pred_valid, yvalid)

        if epoch % print_every == 0:
            print(f"epoch {epoch}: training loss {train_loss.item()}, valid loss {valid_loss.item()}")

        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            num_worse_valid_losses = 0
            # Saving the model with the lowest validation loss
            model.save_model()
        else:
            num_worse_valid_losses += 1

        if num_worse_valid_losses > patience:
            num_worse_valid_losses = 0
            number_of_rounds += 1
            learning_rate *= lr_reduction_factor
            print(f"# Round {number_of_rounds}: reduced lr from {learning_rate/lr_reduction_factor} to {learning_rate}")
            # Resetting the optimizer
            optimizer = torch.optim.Adam(model.get_parameters(), lr=learning_rate)
            # Load the model with the lowest validation loss
            model.load_model()
        else:
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if number_of_rounds >= max_number_of_rounds:
            break




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
