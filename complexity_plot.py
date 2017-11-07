#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

def main(args):
    print("Loading {}...".format(args.filename))

    data = pickle.load(open(args.filename, 'rb'))
    method = data['method']
    complexity = np.array(data['complexity'])
    files = data['files']
    files = [os.path.splitext(os.path.basename(f))[0] for f in files]
    channels = data['channels']

    xinds = np.arange(len(files))
    if args.exclude_files:
        exclude_inds = [files.index(x) for x in args.exclude_files]
        xinds = np.setdiff1d(xinds, exclude_inds)
        complexity = complexity[xinds]
        files = [files[i] for i in xinds]

    n_results = len(complexity)
    n_channels = np.mean([len(chs) for chs in channels])

    print("  {} results, {:.2f} channels each".format(n_results, n_channels))
    if channels[1:] == channels[:-1]:
        print("  Channels: {}".format(channels[0]))
    else:
        print("  Channels: mixed")
    print("  Method: {}".format(method))
    print("  #samples: {}".format(data['nsamples']))
    print("  Compression: {}".format(data['compression']))
    print("")

    for i, c in enumerate(complexity):
        if method == 'per-channel':
            mean = np.mean(c)
            std = np.std(c)
            ch_sort = np.argsort(c)
            c_sort = np.sort(c)
            print("  [{}] {}: mean={:.2f} std={:.2f} min={:.2f} max={:.2f}".format(
                i, files[i], mean, std, c_sort[0], c_sort[-1]))
            print("       c_sort=[{}]".format(
                ", ".join("{:.2f}".format(ci) for ci in c_sort)))
            print("       ch_sort={}".format(ch_sort.tolist()))
        else:
            print("  [{}] {}: {}".format(i, files[i], c))

    if args.style:
        plt.style.use(args.style)

    if args.title is None:
        title = os.path.basename(args.filename)
    else:
        title = args.title

    plt.title(title)

    x = np.arange(len(complexity))
    if args.dates:
        x = data['dates']

    if method == 'per-channel':
        # Per channel scatter plot
        plot = args.plot.split(',')
        if 'all' in plot:
            plot = ['mean', 'std', 'max', 'scatter']

        line_color = None

        for p in plot:
            mean = np.array([np.mean(v) for v in complexity])
            std = np.array([np.std(v) for v in complexity])
            max = np.array([np.max(v) for v in complexity])

            if p == 'mean':
                line, = plt.plot(x, mean)
                line_color = line.get_color()
            elif p == 'max':
                line, = plt.plot(x, max)
            elif p == 'std':
                line = plt.fill_between(x, mean-std, mean+std, alpha=0.25, facecolor=line_color)
                line_color = line.get_facecolor()
            elif p == 'scatter':
                color = None
                for i, v in enumerate(complexity):
                    line, = plt.plot(np.repeat(i, len(v)), v, '.', color=color)
                    color = line.get_color()
            else:
                raise ValueError(p)

    else:
        plt.plot(x, complexity)

    # ugh...
    if args.xticks:
        labels = args.xticks.split(',')
        xticks = np.linspace(x[0], x[-1], len(labels))
        plt.xticks(xticks, labels)
    elif not args.dates:
        plt.xticks(x, files, rotation='vertical')
        plt.subplots_adjust(bottom=0.20)

    ymin, ymax = plt.ylim()
    if ymin < 0:
        plt.ylim(ymin=0)
    plt.ylabel("Complexity")
    plt.xlabel("Time")

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Complexity plot')
    parser.add_argument('-o', '--output', metavar='FILE',
                        help='save plot to file')
    parser.add_argument('-p', '--plot', default='scatter,mean,std',
                        help='what to plot [mean|std|max|scatter] (default: %(default)s)')
    parser.add_argument('-d', '--dates', action='store_true',
                        help='plot dates on x axis')
    parser.add_argument('--xticks', help='xtick labels')
    parser.add_argument('-x', '--exclude-files', nargs='*', help='exclude files from plot')
    parser.add_argument('-t', '--title')
    parser.add_argument('--style', default=None)
    parser.add_argument('filename', metavar='FILE',
                        help='pickle file from complexity_analysis.py')

    args = parser.parse_args()
    main(args)
