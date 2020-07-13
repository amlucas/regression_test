#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import argparse
import numpy as np
import random
import torch

from surrogate_model import Surrogate

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('surrogate_dir', type=str, help="Directory that contains the trained surrogate files.")
    parser.add_argument('inout_data', type=str, help='.pt file that contains the data that has been trained on.')
    parser.add_argument('--seed', type=int, default=42, help='Parameters are chosen randomly (within the train data) from this seed.')
    parser.add_argument('--n', type=int, default=1, help='Number of parameter sets to view.')
    args = parser.parse_args(argv)

    surr = Surrogate(args.surrogate_dir)
    random.seed(args.seed)

    inout_data = torch.load(args.inout_data)
    inputs = inout_data["inputs"]
    outputs = inout_data["outputs"]
    input_varnames = inout_data["input_varnames"]

    nin, ndata = inputs.shape

    times = inputs[input_varnames.index('time')]
    tmax = np.max(times)
    ntimes = int(tmax+1)

    for params_try in range(args.n):

        pid = random.randint(0, ndata-1)
        pid = (pid // ntimes) * ntimes

        x = inputs[:,pid:pid+ntimes]
        x = x.transpose()
        t = x[:,-1]

        y = surr.evaluate(x.tolist())
        channels = surr.get_out_channels()

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.subplots()

        for i, channel in enumerate(channels):
            ax.plot(t, y[:,i], label = f"{channel}: surrogate")
            ax.plot(t, outputs[i,pid:pid+ntimes], '+', label = f"{channel}: simulation")
        ax.set_xlabel("t")
        ax.set_ylabel("daily reported")

        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
