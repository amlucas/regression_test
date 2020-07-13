#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np

def sir(*,
        beta: float,
        gamma: float,
        I0: float,
        N: float=1e6,
        tmax: float=100):
    from scipy.integrate import odeint

    def rhs(x, t):
        S, I, R = x
        return [
            -beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I
        ]

    S0 = N-I0
    R0 = 0
    x0 = [S0, I0, R0]
    times = np.arange(tmax)
    x = odeint(rhs, x0, times)

    I = x[:,1]
    return times, I

def main(argv):
    gamma = 0.2
    beta = 2 * gamma
    N = 1e6
    I0 = N * 0.01
    t, I = sir(beta=beta, gamma=gamma, I0=I0, N=N)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(t, I, '-+')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$I$")
    plt.show()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
