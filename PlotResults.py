import argparse
import json


def plot_the_stuff(file_path):
    with open(file_path, "r") as fin:
        file_lines = fin.readlines()
    
    note = file_lines[0]
    data = [json.loads(line) for line in file_lines[1:]]
    ############################################################todo: start plotting!!!





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path',  help='path to the output file.')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    plot_the_stuff(args.path)