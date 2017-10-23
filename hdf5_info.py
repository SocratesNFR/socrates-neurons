#!/usr/bin/env python2
from __future__ import print_function, division
import sys
import numpy as np

import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg

McsPy.McsData.VERBOSE = False

def split_where(mask):
    mask = np.concatenate(([False], mask, [False] ))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    return idx.reshape((-1, 2))

def hdf5_info(filename, channel_list=False):
    print("{}:".format(filename))
    raw_data = McsPy.McsData.RawData(filename)
    for i, rec in raw_data.recordings.items():
        print("  Recording {}:".format(i))
        for j, stream in rec.analog_streams.items():
            n_channels = stream.channel_data.shape[0]
            n_samples = stream.channel_data.shape[1]
            print("    Stream {}: {} channels, {} samples".format(j, n_channels, n_samples))
            if channel_list:
                # channels = stream.channel_infos.keys()
                print("      Channels:")
                for k, (ch, ci) in enumerate(stream.channel_infos.items()):
                    if k % 8 == 0:
                        print("      ", end="")
                    # ci = stream.channel_infos[ch]
                    s = "[{}] \"{}\"".format(ch, ci.label)
                    print(s.ljust(12), end="")
                    if k % 8 == 7:
                        print("")
                print("\n")


def main(args):
    for filename in args.files:
        hdf5_info(filename, args.list)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='+',
            help='hdf5 files')
    parser.add_argument('-l', '--list', action='store_true',
            help='list available channels')
    parser.add_argument('-r', '--recording', type=int, default=0,
            help='recording id')

    args = parser.parse_args()
    main(args)
