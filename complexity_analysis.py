#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import zlib
import bz2
from collections import defaultdict
import pickle
import signal
import pdb

plt.style.use('ggplot')

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg

McsPy.McsData.VERBOSE = False

def handle_pdb(sig, frame):
    pdb.Pdb().set_trace(frame)    

def install_pdb(sig = signal.SIGINT):
    signal.signal(sig, handle_pdb)

install_pdb()

def digitize(data, th_lo, th_hi):
    return np.where((data > th_lo) & (data < th_hi), 0, 1).astype('uint8')

compress = bz2.compress

def complexity_all_channels(ddata):
    return len(compress(ddata.tostring())) / len(ddata)

def complexity_per_channel(ddata):
    return list(map(complexity_all_channels, ddata.T))

def main(args):
    global compress

    complexity = []
    channel_labels = []
    channel_list = []

    if args.compression == 'zlib':
        compress = zlib.compress
    else:
        compress = bz2.compress

    if args.channels != 'all':
        channels = map(int, args.channels)
    else:
        channels = None

    for i, filename in enumerate(args.files):
        print("Processing {}...".format(filename))
        raw_data = McsPy.McsData.RawData(filename)
        rec = raw_data.recordings[0]
        stream = rec.analog_streams[0]

        channel_data = stream.channel_data
        if args.nsamples != 'all':
            # channel_data = channel_data[:,:231000] # For testing
            assert args.nsamples <= channel_data.shape[1], "Not enough samples (stream has {})".format(channel_data.shape[1])
            channel_data = channel_data[:,:args.nsamples]

        chs = channels
        if not chs:
            chs = stream.channel_infos.keys()

        channel_list.append(chs)

        print("  {} channels, {} samples".format(channel_data.shape[0], channel_data.shape[1]))
        print("  ", end='')

        ddata = []
        channel_labels.append([stream.channel_infos[ch].label for ch in chs])
        for j, ch in enumerate(chs):
            print(".", end='')
            sys.stdout.flush()

            row_index = stream.channel_infos[ch].row_index
            data = channel_data[row_index]
            mean = np.mean(data)
            std = np.std(data)
            th_lo = mean - 5 * std
            th_hi = mean + 5 * std
            dd = digitize(data, th_lo, th_hi)
            ddata.append(dd)

        # Order by time
        ddata = np.array(ddata).T

        print("")

        if args.method == 'per-channel':
            oc = complexity_per_channel(ddata)
            print("  Complexity (mean):", np.mean(oc))
        else:
            oc = complexity_all_channels(ddata)
            print("  Complexity:", oc)

        # print("  channel_data:", channel_data.shape, channel_data.dtype)
        # print("  ddata:", ddata.shape, ddata.dtype)
        # print("  channel_labels:", channel_labels)

        complexity.append(oc)


    # print("complexity=", complexity)

    print("")
    print("Done!")
    print("")
    print("Complexity:")

    for i, c in enumerate(complexity):
        if args.method == 'per-channel':
            mean = np.mean(c)
            std = np.std(c)
            print("{}: mean={} std={}".format(i, mean, std))
        else:
            print("{}: {}".format(i, c))

    if args.output:
        print("")
        print("Saving results to {}...".format(args.output))
        d = {
            'files': args.files,
            'method': args.method,
            'compression': args.compression,
            'channels': channel_list,
            'channel_labels': channel_labels,
            'nsamples': args.nsamples,
            'complexity': complexity
        }
        pickle.dump(d, open(args.output, 'wb'))





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Complexity analysis')
    parser.add_argument('-o', '--output', metavar='FILE',
            help='save result to file (pickle)')
    parser.add_argument('-c', '--compression', choices=('zlib', 'bz2'), default='bz2',
            help='compression algorithm (default: %(default)s)')
    parser.add_argument('-m', '--method', choices=('all', 'per-channel'), default='all',
            help='complexity method')
    parser.add_argument('-ch', '--channels', nargs='*', default='all')
    parser.add_argument('-n', '--nsamples', default='all')
    parser.add_argument('files', nargs='+', metavar='FILE',
            help='files to analyse (hdf5)')

    args = parser.parse_args()
    main(args)
