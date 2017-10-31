#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
from collections import defaultdict
import pickle

from nevroutil import *
from hdf5_stats import stats_available

inf = float('inf')

def normalize_stats(stats_data, divisor):
    if len(stats_data.shape) > len(divisor.shape):
        divisor = divisor.reshape((-1, 1))
    return stats_data / divisor


def plot_stats_total(stats, stats_data, xticks, labels, title=None):
    plt.figure(figsize=(11,11))

    x = np.array(xticks)
    for i, k in enumerate(stats):
        y = stats_data[k]
        if len(y.shape) == 2:
            # Aggregate
            if k in ('mean', 'std'):
                y = np.mean(y, axis=1)
            else:
                y = np.sum(y, axis=1)

        line, = plt.plot(x, y, label=labels[i])

    plt.legend()
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.title(title)

def plot_stats_per_channel(channels, stats, stats_data, xticks, labels, title=None, heatmap=False):
    n_stats = len(stats)
    n_cols = int(np.ceil(np.sqrt(n_stats)))
    n_rows = int(np.ceil(n_stats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11,11))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(xticks))
    for i, k in enumerate(stats):
        y = stats_data[k]
        ax = axes[i]
        ax.set_title(labels[i])
        if len(y.shape) == 2:
            if heatmap:
                Z = y.T
                Y = np.arange(len(channels)+1)

                im = ax.imshow(Z, aspect='auto', origin='lower')

                y_ax = ax.get_yaxis()  ## Get X axis
                y_ax.set_major_locator(ticker.MaxNLocator(integer=True))  ## Set major locators to integer values


                yticks = ax.get_yticks().astype(int)
                yticklabels = [channels[j] if j >= 0 and j < len(channels) else 0 for j in yticks]
                ax.set_yticklabels(yticklabels)

                ax.grid(False)
                ax.set_ylabel("Channel")

                def format_coord(x, y):
                    s = ""
                    xi = int(x+0.5)
                    if xi >= 0 and xi < len(xticks):
                        # s += str(x)
                        s += xticks[xi]

                    yi = int(y+0.5)
                    if yi >= 0 and yi < len(channels):
                        s += "\tCH{}".format(channels[yi])

                    return s

                ax.format_coord = format_coord

                cb = plt.colorbar(im)
                cb.set_label(labels[i])

            else:
                for j, ch in enumerate(channels):
                    ax.plot(x, y.T[j], label="CH{}".format(ch))


        else:
            ax.plot(x, y)

        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation='vertical')
        # ax.legend()
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.subplots_adjust(bottom=0.4)
    plt.suptitle(title)


def main(args):
    print("Loading {}...".format(args.file))

    data = pickle.load(open(args.file, 'rb'))
    stats_data = data['stats']
    stats_available = set(stats_data.keys())
    channels_available = data['channels']
    t0 = data['t0']
    t1 = data['t1']
    xticks = list(map(os.path.basename, data['files']))

    print("Stats available:", ", ".join(stats_available))

    if args.stats == 'all':
        stats = stats_available
    else:
        stats = set(args.stats.split(','))

    print("Stats:", ", ".join(stats))

    assert all(s in stats_data for s in stats)
    if args.normalize:
        assert 'duration' in stats_data

    # Channel selection
    channels = args.channels
    if channels == 'all':
        channels = channels_available
        title = "All channels"
    else:
        channels = channels.split(',')
        channels = list(map(int, channels))
        assert all(ch in channels_available for ch in channels)
        title = "Channels: " + args.channels

    if args.exclude_channels:
        exclude_channels = args.exclude_channels.split(',')
        exclude_channels = list(map(int, exclude_channels))
        channels = list(set(channels) - set(exclude_channels))
        title += " excluding " + args.exclude_channels

    if args.channels != 'all' or args.exclude_channels:
        # Filter
        channel_inds = [channels_available.index(ch) for ch in channels]
        print("Filter", channels_available, channels, channel_inds)
        for k in stats_data:
            print(stats_data[k].shape)
            if len(stats_data[k].shape) == 2:
                stats_data[k] = stats_data[k][:,channel_inds]

    title += ", t={}-{}".format(t0, t1)

    labels = list(stats)

    if args.normalize:
        for i, k in enumerate(stats):
            if k not in ('spike_count', 'bit_count'):
                continue
            stats_data[k] = normalize_stats(stats_data[k], stats_data['duration'])
            labels[i] += ' / second'

    if args.mode == 'total':
        plot_stats_total(stats, stats_data, xticks, labels, title)
    else:
        # per-channel
        plot_stats_per_channel(channels, stats, stats_data, xticks, labels, title, args.heatmap)

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot hdf5 stats.')
    parser.add_argument('-o', '--output', metavar='FILE',
            help='save plot to file')
    parser.add_argument('-s', '--stats', default='all',
                        help='what stats to plot [{}] or "all" (default: %(default)s)'.format(','.join(stats_available)))
    parser.add_argument('-ch', '--channels', default='all',
            help='list of channels (default: %(default)s)')
    parser.add_argument('-ech', '--exclude-channels')
    parser.add_argument('-n', '--normalize', action='store_true',
            help='normalize by duration')
    parser.add_argument('-m', '--mode', choices=('total', 'per-channel'), default='total')
    parser.add_argument('-hm', '--heatmap', action='store_true', default=False)
    parser.add_argument('file', help='pickle file from hdf5_stats.py')

    args = parser.parse_args()
    main(args)
