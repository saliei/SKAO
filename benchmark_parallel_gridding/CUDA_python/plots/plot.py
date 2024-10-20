#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

def v1():
    v1_labels = [r"\texttt{gridding}", r"\texttt{gridding\_kernel}"]
    times_v1 = [1.444, 0.005]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v1_labels, times_v1, width=0.3, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 1: CUDA}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.set_yscale("log")
    ax.text(1.1, max(times_v1)-0.04, r'\texttt{gridding:} 1.444', ha='right', va='center', fontsize=10, color='black')
    ax.text(1.1, max(times_v1)-0.52, r'\texttt{gridding\_kernel:} 0.005', ha='right', va='center', fontsize=10, color='black')

    plt.savefig("v1.png")


def v2():
    v2_labels = [r"\texttt{gridding/gridding\_kernel (NP=1)}", r"\texttt{gridding/gridding\_kernel (NP=2)}"]
    times_v2_gridding = [1.526, 1.608] # np=1,2
    times_v2_gridding_kernel = [0.005, 0.003] # np=1,2

    bar_width = 0.30
    index = np.arange(len(v2_labels))


    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars_np1 = ax.bar(index, times_v2_gridding, width=bar_width, edgecolor="black", linewidth=1, color="dimgray")
    bars_np2 = ax.bar(index + bar_width, times_v2_gridding_kernel, width=bar_width, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 2: CUDA/MPI}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.set_yscale("log")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(v2_labels)

    plt.savefig("v2.png")


def get_plot(func, *args):
    plot = funcs_dict.get(func)(*args)
    return plot

if __name__ == "__main__":
    funcs_dict = {"v1": v1,
                  "v2": v2,
                  }
    funcs_list = list(funcs_dict.keys())

    plot_name = sys.argv[1]
    if plot_name not in funcs_list:
        print("plot name should be one of: {}".format(', '.join(funcs_list)))
        sys.exit(1)

    get_plot(plot_name)

