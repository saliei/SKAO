#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

def original():
    version_original = [r"\texttt{opening\_dataset}", r"\texttt{gridding}", r"\texttt{fourier_transform}"]
    times_original = [0.0196, 232.0669, 0.3949]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(version_original, times_original, width=0.2, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Original Version}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.set_yscale("log")
    ax.text(-0.1, max(times_original)-100, r'\texttt{opening\_dataset:} 0.02',   ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.1, max(times_original)-168, r'\texttt{gridding:} 232.06',         ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.1, max(times_original)-200, r'\texttt{fourier\_transform:} 0.40', ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v0_original.png")

def v1tov5():
    versions_no_dist = [r"\texttt{v1}", r"\texttt{v2}", r"\texttt{v3}", r"\texttt{v4}", r"\texttt{v5}"]
    times_no_dist = [66.40, 1.75, 23.63, 26.77, 2.09]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(versions_no_dist, times_no_dist, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Versions 1 to 5}")
    ax.set_xlabel(r"\rmfamily{Gridding Function Version}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.text(0.4, max(times_no_dist)-1, r'\texttt{v1:} 66.40',  ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-6, r'\texttt{v2:} 1.75',   ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-11, r'\texttt{v3:} 23.63', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-16, r'\texttt{v4:} 26.77', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-21, r'\texttt{v5:} 2.09',  ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v1tov5.png")


def v6():
    v6_nworkers = ["1", "2", "4", "8", "16"]
    times_v6 = [2.16, 1.68, 1.61, 1.99, 3.34]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v6_nworkers, times_v6, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 6: Multithreaded}")
    ax.set_xlabel(r"\rmfamily{Number of Threads}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.text(-0.21, max(times_v6)-0.1, r'1 \ Ts: 2.16', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.3, r'2 \ Ts: 1.68', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.5, r'4 \ Ts: 1.61', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.7, r'8 \ Ts: 1.99', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.9, r'16 Ts: 3.34',  ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v6.png")


def v7v8():
    v7_nmpi = ["1", "2", "4", "8", "16"]
    v8_nmpi = ["1", "2", "4", "8", "16"]
    times_v7 = [94.56, 18.37, 6.28, 3.84, 3.61]
    times_v8 = [34.29, 24.25, 11.65, 9.02, 11.38]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    bars1 = ax1.bar(v7_nmpi, times_v7, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax1.set_title(r"\rmfamily{Version 7: MPI/Timsteps}")
    ax1.set_xlabel(r"\rmfamily{Number of MPI Processes}")
    ax1.set_ylabel(r"\rmfamily{Time(s)}")
    ax1.text(3.2, max(times_v7)-2, r'1 \ Ps: 94.56', ha='left', va='center', fontsize=10, color='black')
    ax1.text(3.2, max(times_v7)-7, r'2 \ Ps: 18.37', ha='left', va='center', fontsize=10, color='black')
    ax1.text(3.2, max(times_v7)-13, r'4 \ Ps: 6.28', ha='left', va='center', fontsize=10, color='black')
    ax1.text(3.2, max(times_v7)-19, r'8 \ Ps: 3.84', ha='left', va='center', fontsize=10, color='black')
    ax1.text(3.2, max(times_v7)-25, r'16 Ps: 3.61',  ha='left', va='center', fontsize=10, color='black')

    bars2 = ax2.bar(v8_nmpi, times_v8, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax2.set_title(r"\rmfamily{Version 8: MPI/Baselines}")
    ax2.set_xlabel(r"\rmfamily{Number of MPI Processes}")
    ax2.set_ylabel(r"\rmfamily{Time(s)}")
    ax2.text(3.2, max(times_v8)-1, r'1 \ Ps: 34.29', ha='left', va='center', fontsize=10, color='black')
    ax2.text(3.2, max(times_v8)-3, r'2 \ Ps: 24.25', ha='left', va='center', fontsize=10, color='black')
    ax2.text(3.2, max(times_v8)-5, r'4 \ Ps: 11.65', ha='left', va='center', fontsize=10, color='black')
    ax2.text(3.2, max(times_v8)-7, r'8 \ Ps: 9.02',  ha='left', va='center', fontsize=10, color='black')
    ax2.text(3.2, max(times_v8)-9, r'16 Ps: 11.38',  ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig("v7v8.png")


def get_plot(func, *args):
    plot = funcs_dict.get(func)(*args)
    return plot

if __name__ == "__main__":
    funcs_dict = {"original": original,
                  "v1tov5": v1tov5,
                  "v6": v6,
                  "v7v8": v7v8,
                  }
    funcs_list = list(funcs_dict.keys())

    plot_name = sys.argv[1]
    if plot_name not in funcs_list:
        print("plot name should be one of: {}".format(', '.join(funcs_list)))
        sys.exit(1)

    get_plot(plot_name)

