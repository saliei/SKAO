#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

def v1():
    v1_nomp = ["1", "2", "4", "8", "16", "32", "64", "128"]
    times_v1 = [0.316, 0.172, 0.094, 0.053, 0.035, 0.034, 0.037, 0.048]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v1_nomp, times_v1, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 1: OpenMP}")
    ax.set_xlabel(r"\rmfamily{Number of Threads}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    plt.savefig("v1.png")


def v2():
    v2_nmpi = ["1", "2", "4", "8", "16", "32", "64", "128"]
    times_v2 = [0.340, 0.198, 0.181, 0.170, 0.198, 0.306, 0.386, 0.674]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v2_nmpi, times_v2, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 2: MPI}")
    ax.set_xlabel(r"\rmfamily{Number of MPI Processes}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")

    plt.savefig("v2.png")

def v3():
    v3_nomp_simd = ["1", "2", "4", "8", "16", "32", "64", "128"]
    times_v3 = [0.162, 0.088, 0.045, 0.030, 0.026, 0.030, 0.034, 0.045]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v3_nomp_simd, times_v3, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 3: SIMD/OpenMP}")
    ax.set_xlabel(r"\rmfamily{Number of Threads}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")

    plt.savefig("v3.png")

def v4():
    # 16 threads, num mpi
    v4_nmpi_t16 = ["1", "2", "4", "8"]
    times_t16 = [0.196, 0.127, 0.131, 0.150]

    # 8 threads, num mpi
    v4_nmpi_t8 = ["1", "2", "4", "8", "16"]
    times_t8 = [0.195, 0.127, 0.140, 0.151, 0.312]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    bars1 = ax1.bar(v4_nmpi_t16, times_t16, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    bars2 = ax2.bar(v4_nmpi_t8, times_t8, width=0.4, edgecolor="black", linewidth=1, color="dimgray")

    ax1.set_title(r"\rmfamily{Version 4: SIMD/MPI/OpenMP - 16 Threads}")
    ax1.set_xlabel(r"\rmfamily{Number of MPI Processes}")
    ax1.set_ylabel(r"\rmfamily{Time(s)}")

    ax2.set_title(r"\rmfamily{Version 4: SIMD/MPI/OpenMP - 8 Threads}")
    ax2.set_xlabel(r"\rmfamily{Number of MPI Processes}")
    ax2.set_ylabel(r"\rmfamily{Time(s)}")

    plt.tight_layout()
    plt.savefig("v4.png")


def get_plot(func, *args):
    plot = funcs_dict.get(func)(*args)
    return plot

if __name__ == "__main__":
    funcs_dict = {"v1": v1,
                  "v2": v2,
                  "v3": v3,
                  "v4": v4
                  }
    funcs_list = list(funcs_dict.keys())

    plot_name = sys.argv[1]
    if plot_name not in funcs_list:
        print("plot name should be one of: {}".format(', '.join(funcs_list)))
        sys.exit(1)

    get_plot(plot_name)

