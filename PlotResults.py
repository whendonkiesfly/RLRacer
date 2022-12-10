import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_the_stuff(file_path, title):
    with open(file_path, "r") as fin:
        file_lines = fin.readlines()
    
    note = file_lines[0]
    data_entries = [json.loads(line) for line in file_lines[1:]]

    fig, ax1 = plt.subplots()
    y_vals = list(entry["missing_checkpoints"] for entry in data_entries)
    x_vals = list(range(len(data_entries)))
    l1, = ax1.plot(x_vals, y_vals, label="Missed Checkpoints", color="black")
    ax1.set(xlabel="Race #", ylabel="Missed Checkpoints", title=title)

    #calculate equation for trendline
    z = np.polyfit(x_vals, y_vals, 2)
    p = np.poly1d(z)
    l2, = ax1.plot(x_vals, p(x_vals), label="Missed Checkpoint Trendline", color="blue")
    

    ax2 = ax1.twinx()
    ax2.set(ylabel="Race Time")
    x_vals = list(i for i in range(len(data_entries)) if data_entries[i]["missing_checkpoints"] == 0)
    y_vals = list(data_entries[i]["race_time"] for i in x_vals)
    l3= ax2.scatter(x_vals, y_vals, label="Race Times", color="orange")

    #calculate equation for trendline
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    l4, = ax2.plot(x_vals, p(x_vals), label="Race Time Trendline", color="green")
    

    ax1.legend([l1, l2, l3, l4], ["Missed Checkpoints", "Missed Checkpoint Trend", "Race Times", "Race Time Trend"])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path',  help='Path to the output file.')
    parser.add_argument('title',  help='Plot title')

    args = parser.parse_args()
    plot_the_stuff(args.path, args.title)