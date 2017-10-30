#!/usr/bin/env python2
#
# Stats:
#   sample_count
#   spike_count
#   bit_count
#   mean
#   std
#
# Aggregate:
#   All channels
#   Per channel
#
# Filters:
#   Channels
#   Time range
#
# Data format
#   array with N rows, M columns
#      N = number of files
#      M = number of channels
#
#   -- or --
#
#   array with N rows
#      N = number of files
#
from __future__ import print_function, division
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
from collections import defaultdict

from nevroutil import *

inf = float('inf')

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg

McsPy.McsData.VERBOSE = False

stats_available = set(('sample_count', 'duration', 'spike_count', 'bit_count', 'mean', 'std', 'min', 'max'))

def sample_count(raw_data):
    """ Sample count all channels """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    n_samples = stream.channel_data.shape[1]

    return n_samples

def duration(raw_data):
    """ Recording duration in seconds """
    rec = raw_data.recordings[0]
    return rec.duration_time.to('seconds').magnitude

def spike_count(raw_data, channels, t0=0, t1=inf):
    """ Spike count per channel """
    rec = raw_data.recordings[0]
    te = rec.timestamp_streams[0].timestamp_entity
    timestamps = get_timestamp_data_in_range(te, channels, t0, t1)

    spike_counts = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in timestamps:
            spike_counts[i] = len(timestamps[ch])

    return spike_counts

def bit_count(raw_data, channels, t0=0, t1=inf):
    """ Bit count per channel """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    stream_data = get_stream_data_in_range(stream, channels, t0, t1)

    bit_counts = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in stream_data:
            data = stream_data[ch]

            mean = np.mean(data)
            std = np.std(data)
            th_lo = mean - 5 * std
            th_hi = mean + 5 * std

            bits = digitize(data, th_lo, th_hi)
            idx = split_where(bits)
            bit_counts[i] = len(idx)

    return bit_counts

def stream_mean(raw_data, channels, t0=0, t1=inf):
    """ Mean per channel """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    stream_data = get_stream_data_in_range(stream, channels, t0, t1)

    means = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in stream_data:
            data = stream_data[ch]
            means[i] = np.mean(data)

    return means

def stream_std(raw_data, channels, t0=0, t1=inf):
    """ Std per channel """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    stream_data = get_stream_data_in_range(stream, channels, t0, t1)

    stds = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in stream_data:
            data = stream_data[ch]
            stds[i] = np.std(data)

    return stds

def stream_min(raw_data, channels, t0=0, t1=inf):
    """ Min per channel """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    stream_data = get_stream_data_in_range(stream, channels, t0, t1)

    mins = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in stream_data:
            data = stream_data[ch]
            mins[i] = np.min(data)

    return mins

def stream_max(raw_data, channels, t0=0, t1=inf):
    """ Max per channel """
    rec = raw_data.recordings[0]
    stream = rec.analog_streams[0]
    stream_data = get_stream_data_in_range(stream, channels, t0, t1)

    maxs = np.zeros(len(channels))
    for i, ch in enumerate(channels):
        if ch in stream_data:
            data = stream_data[ch]
            maxs[i] = np.max(data)

    return maxs

def normalize_stats(stats_data, divisor):
    if len(stats_data.shape) > len(divisor.shape):
        divisor = divisor.reshape((-1, 1))
    return stats_data / divisor




def plot_stats_total(stats, stats_data, xticks, labels, title=None):
    plt.figure(figsize=(11,11))

    x = xticks
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
    channels = np.array(sorted(channels))
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
    stats = args.stats.split(',')
    t0 = args.t0
    t1 = args.t1

    assert set(stats) - stats_available == set()

    stats_data = defaultdict(list)

    # Channel selection
    channels = args.channels
    if channels:
        channels = channels.split(',')
        channels = set(map(int, channels))
        title = "Channels: " + args.channels
    else:
        title = "All channels"
        channels = set(range(60)) # TODO: dont hardcode this

    if args.exclude_channels:
        exclude_channels = args.exclude_channels.split(',')
        exclude_channels = set(map(int, exclude_channels))
        channels = channels - exclude_channels
        title += " excluding " + args.exclude_channels

    title += ", t={}-{}".format(t0, t1)

    for i, filename in enumerate(args.files):
        print("Processing {}...".format(filename))

        raw_data = McsPy.McsData.RawData(filename)
        rec = raw_data.recordings[0]
        stream = rec.analog_streams[0]
        timestamps = rec.timestamp_streams[0].timestamp_entity

        if 'sample_count' in stats:
            sc = sample_count(raw_data)
            stats_data['sample_count'].append(sc)

        if 'duration' in stats or args.normalize:
            d = duration(raw_data)
            stats_data['duration'].append(d)

        if 'spike_count' in stats:
            sc = spike_count(raw_data, channels, t0, t1)
            stats_data['spike_count'].append(sc)

        if 'bit_count' in stats:
            bc = bit_count(raw_data, channels, t0, t1)
            stats_data['bit_count'].append(bc)

        if 'mean' in stats:
            m = stream_mean(raw_data, channels, t0, t1)
            stats_data['mean'].append(m)

        if 'std' in stats:
            m = stream_std(raw_data, channels, t0, t1)
            stats_data['std'].append(m)

        if 'min' in stats:
            m = stream_min(raw_data, channels, t0, t1)
            stats_data['min'].append(m)

        if 'max' in stats:
            m = stream_max(raw_data, channels, t0, t1)
            stats_data['max'].append(m)

        # date = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(raw_data.date_in_clr_ticks)/10)
        # dates.append(date)

    for k in stats_data:
        stats_data[k] = np.array(stats_data[k])

    labels = list(stats)

    if args.normalize:
        for i,k in enumerate(stats):
            if k not in ('spike_count', 'bit_count'):
                continue
            stats_data[k] = normalize_stats(stats_data[k], stats_data['duration'])
            labels[i] += ' / second'

    if args.mode == 'total':
        xticks = list(map(os.path.basename, args.files))
        plot_stats_total(stats, stats_data, xticks, labels, title)
    else:
        # per-channel
        xticks = list(map(os.path.basename, args.files))
        plot_stats_per_channel(channels, stats, stats_data, xticks, labels, title, args.heatmap)

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot hdf5 data.')
    parser.add_argument('-o', '--output', metavar='FILE',
            help='save plot to file')
    parser.add_argument('-s', '--stats', default='spike_count',
                        help='what stats to plot [{}] (default: %(default)s)'.format(','.join(stats_available)))
    parser.add_argument('-t0', type=float, default=0)
    parser.add_argument('-t1', type=float, default=inf)
    parser.add_argument('-ch', '--channels', help='list of channels (default: all)')
    parser.add_argument('-ech', '--exclude-channels')
    parser.add_argument('-n', '--normalize', action='store_true',
            help='normalize by duration')
    parser.add_argument('-m', '--mode', choices=('total', 'per-channel'), default='total')
    parser.add_argument('-hm', '--heatmap', action='store_true', default=False)
    parser.add_argument('files', nargs='+', metavar='FILE',
            help='files to analyse (hdf5)')

    args = parser.parse_args()
    main(args)
