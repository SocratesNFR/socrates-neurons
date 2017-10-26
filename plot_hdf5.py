#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime

inf = float('inf')

plt.style.use('ggplot')

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg

McsPy.McsData.VERBOSE = False

def split_where(mask):
    mask = np.concatenate(([False], mask, [False] ))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    return idx.reshape((-1, 2))


def print_list(filename):
    raw_data = McsPy.McsData.RawData(filename)
    print("Listing {}".format(filename))

    for i, rec in raw_data.recordings.items():
        print("Recording {}:".format(i))

        assert len(rec.analog_streams) == 1, "More than one analog stream?"
        stream = rec.analog_streams[0]
        n_channels = stream.channel_data.shape[0]
        n_samples = stream.channel_data.shape[1]
        channels = stream.channel_infos.keys()
        # print("  Analog stream: {}".format(stream.label))
        print("  {} channels, {} samples per channel".format(n_channels, n_samples))
        for j, ch in enumerate(channels):
            if j % 8 == 0:
                print("  ", end="")
            ci = stream.channel_infos[ch]
            s = "[{}] \"{}\"".format(ch, ci.label)
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

def format_time(t, pos=None):
    dt = datetime.timedelta(seconds=t)
    return str(dt)

def plot(filename, recording=0, channels=None, spikes=False, t0=0, t1=inf, digitize=False, output=None):
    raw_data = McsPy.McsData.RawData(filename)
    rec = raw_data.recordings[recording]

    stream = rec.analog_streams[0]
    segments = rec.segment_streams[0].segment_entity
    timestamps = rec.timestamp_streams[0].timestamp_entity

    if not channels:
        channels = stream.channel_infos.keys()

    n_channels = len(channels)
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    axes = axes.flatten()

    fig.suptitle(filename)

    for i, ch in enumerate(channels):
        label = stream.channel_infos[ch].label
        label = "CH{}: {}".format(ch, label)
        print(label, end='')

        tick = stream.channel_infos[ch].sampling_tick
        tick = ureg.convert(tick.magnitude, tick.units, "second")

        idx_start = 0
        idx_end = stream.channel_data.shape[1]
        if t1 != inf:
            idx_end = int(t1 / tick)
        if t0 > 0:
            idx_start = int(t0 / tick)

        t, unit = stream.get_channel_sample_timestamps(ch, idx_start, idx_end)
        t = ureg.convert(t, unit, "second")

        data, unit = stream.get_channel_in_range(ch, idx_start, idx_end)
        data = ureg.convert(data, unit, "microvolt")

        ax = axes[i]
        ax.set_title(label)
        if i % n_cols == 0:
            ax.set_ylabel("uV")
        if i >= (n_cols * (n_rows - 1)):
            ax.set_xlabel("Time")

        ax.set_xlim(t[0], t[-1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        ax.plot(t, data, color='#E24A33')

        ymin, ymax = ax.get_ylim()
        ylim = max(abs(ymin), abs(ymax))
        yheight = 2 * ylim
        ybot = -ylim

        if spikes:
            # Segments
            if ch in segments:
                signal, unit = segments[ch].get_segment_in_range(0, False)
                signal = signal * ureg.convert(1, unit, "microvolt")

                ts, unit = segments[ch].get_segment_sample_timestamps(0, False)
                ts = ureg.convert(ts, unit, "second")

                ts = np.ma.masked_outside(ts, t0, t1)
                signal = np.ma.array(signal, mask=ts.mask)

                cols = np.flatnonzero(np.ma.count(ts, axis=0))
                n_segments = len(cols)
                for j in cols:
                    ax.plot(ts[:,j], signal[:,j], color='#8EBA42')

                print(", {} segments".format(n_segments), end='')

            # Timestamps
            if ch in timestamps:
                ts, unit = timestamps[ch].get_timestamps()
                ts = ureg.convert(ts, "microsecond", "second")
                ts = np.ma.masked_outside(ts, t0, t1)
                ts = ts.compressed()

                ymax = ybot + yheight/20
                ax.vlines(ts, ybot, ymax, color='#8EBA42')
                ybot += 1.1*yheight/20

                print(", {} timestamps".format(len(ts)), end='')

        if digitize:
            # +/- 5std
            mean = np.mean(data)
            std = np.std(data)
            print(", mean={}, std={}, 5*std={}".format(mean, std, 5*std), end='')
            ax.axhline(5*std, color='#E24A33')
            ax.axhline(-5*std, color='#348ABD')

            th_hi = 5*std
            th_lo = -5*std
            bits = np.where((data > th_lo) & (data < th_hi), 0, 1)

            idx = split_where(bits)
            for i, j in idx:
                ax.plot(t[i:j], data[i:j], color='#988ED5')

            print(", detected {} high bits".format(idx.shape[0]), end='')

            height = yheight / 20
            ax.plot(t, ybot + height * bits, color='#988ED5')
            # ax.set_ylim(ymin=ymin)

        ax.set_ylim(-ylim, ylim)

        print("")


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

    plot(args.filename, args.recording, args.channels, args.spikes, args.t0,
            args.t1, args.digitize, args.output)

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
    parser.add_argument('-s', '--spikes', action='store_true',
            help='plot spike data')
    parser.add_argument('-d', '--digitize', action='store_true')
    parser.add_argument('-t0', type=float, default=0)
    parser.add_argument('-t1', type=float, default=inf)
    parser.add_argument('channels', nargs='*', type=int,
            help='list of channels to plot (default: all)')

    args = parser.parse_args()
    main(args)
