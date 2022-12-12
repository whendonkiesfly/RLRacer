import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_datasets(files):
    datasets = []
    for file in files:
        with open(file, "r") as fin:
            datasets.append([json.loads(line) for line in fin.readlines()])
    return datasets


# def load_missing_checkpoints(datasets):
#     missing_checkpoints = []
#     dataset_length = len(datasets[0])
#     for i in range(dataset_length):
#         missing_checkpoints.append(sum(dataset[i]["missing_checkpoint"] for dataset in datasets) / len(datasets))
#     return missing_checkpoints


# def load_race_times(datasets):
#     race_time = []


def load_missing_checkpoints(datasets):
    x_vals = list(range(1, len(datasets[0])+1)) * len(datasets)
    y_vals = []
    for dataset in datasets:
        y_vals.extend([dataset_entry["missing_checkpoints"] for dataset_entry in dataset])
    return x_vals, y_vals

def load_race_times(datasets):
    x_vals = []
    y_vals = []
    for dataset in datasets:
        for i in range(len(dataset)):
            if dataset[i]["missing_checkpoints"] == 0:
                x_vals.append(i)
                y_vals.append(dataset[i]["race_time"])
    return x_vals, y_vals






def plot_vals(ax, x_vals, y_vals, main_color, trend_color, base_label):
    ax.scatter(x_vals, y_vals, label=base_label, color=main_color)###todo: return ret?
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    ax.plot(x_vals, p(x_vals), label=base_label + " Trend", color=trend_color)



TRACK_1_LOG_ROOT = "c:\\temp\\results_track_1"
TRACK_2_LOG_ROOT = "c:\\temp\\results_track_2"


def plot_allvisit_with_vs_without_cp():
    with_cp_files = [os.path.join(TRACK_1_LOG_ROOT, f"allvisitmc_withCP_{i}_results.txt")
                                                                for i in range(3)]
    with_cp_datasets = load_datasets(with_cp_files)

    without_cp_files = [os.path.join(TRACK_1_LOG_ROOT, f"allvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    without_cp_datasets = load_datasets(without_cp_files)

    with_cp_missing_cps = load_missing_checkpoints(with_cp_datasets)
    without_cp_missing_cps = load_missing_checkpoints(without_cp_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="All-Visit Comparison")
    plot_vals(ax1, *with_cp_missing_cps, "green", "lime", "Missing CPs with CP")
    plot_vals(ax1, *without_cp_missing_cps, "blue", "cyan", "Missing CPs without CP")

    with_cp_race_times = load_race_times(with_cp_datasets)
    without_cp_race_times = load_race_times(without_cp_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *with_cp_race_times, "red", "orange", "Race Time with CP")
    plot_vals(ax1, *without_cp_race_times, "black", "gray", "Race Time without CP")
    ax1.legend()

    print("With CP Final Race Time", sum(dataset[-1]["race_time"] for dataset in with_cp_datasets) / len(with_cp_datasets))
    print("Without CP Final Race Time", sum(dataset[-1]["race_time"] for dataset in without_cp_datasets) / len(without_cp_datasets))

    plt.show()


def plot_firstvisit_with_vs_without_cp():
    with_cp_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_withCP_{i}_results.txt")
                                                                for i in range(3)]
    with_cp_datasets = load_datasets(with_cp_files)

    without_cp_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    without_cp_datasets = load_datasets(without_cp_files)

    with_cp_missing_cps = load_missing_checkpoints(with_cp_datasets)
    without_cp_missing_cps = load_missing_checkpoints(without_cp_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="First-Visit Comparison")
    plot_vals(ax1, *with_cp_missing_cps, "green", "lime", "Missing CPs with CP")
    plot_vals(ax1, *without_cp_missing_cps, "blue", "cyan", "Missing CPs without CP")

    with_cp_race_times = load_race_times(with_cp_datasets)
    without_cp_race_times = load_race_times(without_cp_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *with_cp_race_times, "red", "orange", "Race Time with CP")
    plot_vals(ax1, *without_cp_race_times, "black", "gray", "Race Time without CP")
    ax1.legend()

    
    print("With CP Final Race Time", sum(dataset[-1]["race_time"] for dataset in with_cp_datasets) / len(with_cp_datasets))
    print("Without CP Final Race Time", sum(dataset[-1]["race_time"] for dataset in without_cp_datasets) / len(without_cp_datasets))


    plt.show()



def plot_allvisit_vs_firstvisit_nocp():
    firstvisit_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    firstvisit_datasets = load_datasets(firstvisit_files)

    allvisit_files = [os.path.join(TRACK_1_LOG_ROOT, f"allvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    allvisit_datasets = load_datasets(allvisit_files)

    firstvisit_missing_cps = load_missing_checkpoints(firstvisit_datasets)
    allvisit_missing_cps = load_missing_checkpoints(allvisit_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="First-Visit vs. All-Visit")
    plot_vals(ax1, *allvisit_missing_cps, "green", "lime", "All-Visit Missing CP")
    plot_vals(ax1, *firstvisit_missing_cps, "blue", "cyan", "First-Visit Missing CP")

    firstvisit_race_times = load_race_times(firstvisit_datasets)
    allvisit_race_times = load_race_times(allvisit_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *firstvisit_race_times, "red", "orange", "First-Visit Race Time")
    plot_vals(ax1, *allvisit_race_times, "black", "gray", "All-Visit Race Time")
    ax1.legend()

    print("First Visit Final Race Time", sum(dataset[-1]["race_time"] for dataset in firstvisit_datasets) / len(firstvisit_datasets))
    print("All Visit Final Race Time", sum(dataset[-1]["race_time"] for dataset in allvisit_datasets) / len(allvisit_datasets))

    plt.show()


def plot_sarsa_vs_firstvisit_nocp():
    firstvisit_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    firstvisit_datasets = load_datasets(firstvisit_files)

    sarsa_files = [os.path.join(TRACK_1_LOG_ROOT, f"sarsa_noCP_{i}_results.txt")
                                                                for i in range(3)]
    sarsa_datasets = load_datasets(sarsa_files)

    firstvisit_missing_cps = load_missing_checkpoints(firstvisit_datasets)
    sarsa_missing_cps = load_missing_checkpoints(sarsa_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="First-Visit vs. SARSA")
    plot_vals(ax1, *sarsa_missing_cps, "green", "lime", "SARSA Missing CP")
    plot_vals(ax1, *firstvisit_missing_cps, "blue", "cyan", "First-Visit Missing CP")

    firstvisit_race_times = load_race_times(firstvisit_datasets)
    sarsa_race_times = load_race_times(sarsa_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *firstvisit_race_times, "red", "orange", "First-Visit Race Time")
    plot_vals(ax1, *sarsa_race_times, "black", "gray", "SARSA Race Time")
    ax1.legend()


    print("First Visit Final Race Time", sum(dataset[-1]["race_time"] for dataset in firstvisit_datasets) / len(firstvisit_datasets))
    print("SARSA Final Race Time", sum(dataset[-1]["race_time"] for dataset in sarsa_datasets) / len(sarsa_datasets))

    plt.show()


def plot_firstvisit_nocp_vs_relearn_withcp():
    firstvisit_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_noCP_{i}_results.txt")
                                                                for i in range(3)]
    firstvisit_datasets = load_datasets(firstvisit_files)

    relearned_files = [os.path.join(TRACK_1_LOG_ROOT, f"firstvisitmc_withCP_relearn_{i}_results.txt")
                                                                for i in range(3)]
    relearned_datasets = load_datasets(relearned_files)

    firstvisit_missing_cps = load_missing_checkpoints(firstvisit_datasets)
    relearned_missing_cps = load_missing_checkpoints(relearned_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="First-Visit without CP-State vs Relearned with CP-State")
    plot_vals(ax1, *relearned_missing_cps, "green", "lime", "Relearned Missing CP")
    plot_vals(ax1, *firstvisit_missing_cps, "blue", "cyan", "First-Visit Missing CP")

    firstvisit_race_times = load_race_times(firstvisit_datasets)
    relearned_race_times = load_race_times(relearned_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *firstvisit_race_times, "red", "orange", "First-Visit Race Time")
    plot_vals(ax1, *relearned_race_times, "black", "gray", "Relearned Race Time")
    ax1.legend()

    
    print("First Visit Final Race Time", sum(dataset[-1]["race_time"] for dataset in firstvisit_datasets) / len(firstvisit_datasets))
    print("relearned Final Race Time", sum(dataset[-1]["race_time"] for dataset in relearned_datasets) / len(relearned_datasets))

    plt.show()


def plot_track2_scratch_withcp_vs_relearned_withcp():
    from_scratch_files = [os.path.join(TRACK_2_LOG_ROOT, f"firstvisitmc_withCP_{i}_results.txt")
                                                                for i in range(3)]
    from_scratch_datasets = load_datasets(from_scratch_files)

    transfer_files = [os.path.join(TRACK_2_LOG_ROOT, f"firstvisitmc_withCP_transfer_{i}_results.txt")
                                                                for i in range(3)]
    transfer_datasets = load_datasets(transfer_files)

    from_scratch_missing_cps = load_missing_checkpoints(from_scratch_datasets)
    transfer_missing_cps = load_missing_checkpoints(transfer_datasets)

    _, ax1 = plt.subplots()
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title="Track 2 Transfer vs From Scratch")
    plot_vals(ax1, *transfer_missing_cps, "green", "lime", "Transfer Missing CP")
    plot_vals(ax1, *from_scratch_missing_cps, "blue", "cyan", "From Scratch Missing CP")

    from_scratch_race_times = load_race_times(from_scratch_datasets)
    transfer_race_times = load_race_times(transfer_datasets)

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    plot_vals(ax1, *from_scratch_race_times, "red", "orange", "From Scratch Race Time")
    plot_vals(ax1, *transfer_race_times, "black", "gray", "Transfer Race Time")
    ax1.legend()

    print("From Scratch Final Race Time", sum(dataset[-1]["race_time"] for dataset in from_scratch_datasets) / len(from_scratch_datasets))
    print("Transfer Final Race Time", sum(dataset[-1]["race_time"] for dataset in transfer_datasets) / len(transfer_datasets))

    plt.show()











if __name__ == "__main__":



    # plot_allvisit_with_vs_without_cp()
    # plot_allvisit_vs_firstvisit_nocp()
    # plot_sarsa_vs_firstvisit_nocp()
    # plot_firstvisit_nocp_vs_relearn_withcp()
    plot_track2_scratch_withcp_vs_relearned_withcp()


