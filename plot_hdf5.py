#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_

McsPy.McsData.VERBOSE = False

def print_list(filename):
    raw_data = McsPy.McsData.RawData(filename)
    print("Listing {}".format(filename))

    for i, rec in raw_data.recordings.items():
        print("Recording {}:".format(i))

        assert len(rec.analog_streams) == 1, "More than one analog stream?"
        stream = rec.analog_streams[0]
        n_channels = stream.channel_data.shape[0]
        n_samples = stream.channel_data.shape[1]
        channel_ids = range(n_channels)
        # print("  Analog stream: {}".format(stream.label))
        print("  {} channels, {} samples per channel".format(n_channels, n_samples))
        for j in range(n_channels):
            if j % 8 == 0:
                print("  ", end="")
            ci = stream.channel_infos[j]
            s = "[{}] \"{}\"".format(j, ci.label)
            print(s.ljust(12), end="")
            if j % 8 == 7:
                print("")
        print("\n")

        '''
        assert len(rec.segment_streams) == 1, "More than one segment stream?"
        stream = rec.segment_streams[0]
        segments = stream.segment_entity
        n_channels = len(segments)
        print("  Segment stream: {}".format(stream.label))
        print("    {} channels".format(n_channels))
        for j, seg in segments.items():
            ci = seg.info
            print("    #{} \"{}\": {} segments".format(
                j, ci.label, seg.segment_sample_count))
        '''


def plot(filename, recording=0, channels=None, output=None):
    raw_data = McsPy.McsData.RawData(filename)
    rec = raw_data.recordings[recording]

    stream = rec.analog_streams[0]
    if not channels:
        channels = range(stream.channel_data.shape[0])

    n_channels = len(channels)
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle(filename)

    t0 = 0
    for i, ch in enumerate(channels):
        label = stream.channel_infos[i].label
        label = "CH{}: {}".format(ch, label)
        print(label)

        t1 = stream.channel_data.shape[1]
        time = stream.get_channel_sample_timestamps(ch, t0, t1)
        # scale time to msecs
        scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
        t = time[0] * scale_factor_for_second# * 1e3

        data, unit = stream.get_channel_in_range(ch, t0, t1)
        # unit is Volt, convert to uV
        data *= 1e6


        ax = axes[i]
        ax.set_title(label)
        if i % n_cols == 0:
            ax.set_ylabel("uV")
        if i >= (n_cols * (n_rows - 1)):
            ax.set_xlabel("s")
        # ax[i].set_ylim(-100, 100)
        ax.set_xlim(t[0], t[-1])
        ax.plot(t, data)

    for i in range(n_channels, axes.shape[0]):
        axes[i].set_visible(False)

    if output:
        plt.savefig(output)
    else:
        plt.show()

def main(args):

    if args.list:
        print_list(args.filename)
        return

    plot(args.filename, args.recording, args.channels, args.output)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot hdf5 data.')
    parser.add_argument('filename', metavar='FILE',
            help='hdf5 file')
    parser.add_argument('-o', '--output', metavar='FILE',
            help='save plot to file')
    parser.add_argument('-l', '--list', action='store_true',
            help='list available channels')
    parser.add_argument('-r', '--recording', type=int, default=0,
            help='recording id')
    parser.add_argument('channels', nargs='*', type=int,
            help='list of channels to plot (default: all)')

    args = parser.parse_args()
    main(args)
