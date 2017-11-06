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
import pickle

from utils import *

inf = float('inf')

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg

McsPy.McsData.VERBOSE = False

stats_available = ['sample_count', 'duration', 'spike_count', 'bit_count', 'mean', 'std', 'min', 'max']

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


def main(args):
    t0 = args.t0
    t1 = args.t1

    if args.stats == 'all':
        stats = stats_available
    else:
        stats = args.stats.split(',')

    assert all(s in stats_available for s in stats)

    # Channel selection
    channels = args.channels
    if channels == 'all':
        channels = list(range(60)) # TODO: dont hardcode this
    else:
        channels = channels.split(',')
        channels = sorted(map(int, channels))

    print("Stats:", stats)
    print("Channels:", args.channels)
    print("Time range: {}-{}".format(t0, t1))

    stats_data = defaultdict(list)
    dates = []

    for i, filename in enumerate(args.files):
        print("Processing {}...".format(filename))

        raw_data = McsPy.McsData.RawData(filename)
        rec = raw_data.recordings[0]
        stream = rec.analog_streams[0]
        timestamps = rec.timestamp_streams[0].timestamp_entity

        if 'sample_count' in stats:
            sc = sample_count(raw_data)
            stats_data['sample_count'].append(sc)

        if 'duration' in stats:
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

        date = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(raw_data.date_in_clr_ticks)/10)
        dates.append(date)

    for k in stats_data:
        stats_data[k] = np.array(stats_data[k])

    print("Saving stats to {}...".format(args.output))

    data = {
        'files': args.files,
        'file_dates': dates,
        'stats': stats_data,
        'channels': channels,
        't0': t0, 't1': t1
    }

    pickle.dump(data, open(args.output, 'wb'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect stats from hdf5 files.')
    parser.add_argument('-o', '--output', metavar='FILE', required=True,
            help='save stats to file')
    parser.add_argument('-s', '--stats', default='all',
                        help='what stats to plot [{}] or "all" (default: %(default)s)'.format(','.join(stats_available)))
    parser.add_argument('-t0', type=float, default=0)
    parser.add_argument('-t1', type=float, default=inf)
    parser.add_argument('-ch', '--channels', default='all',
            help='list of channels (default: %(default)s)')
    parser.add_argument('files', nargs='*', metavar='FILE',
            help='hdf5 files to analyse')

    args = parser.parse_args()
    main(args)
