from __future__ import print_function, division
import numpy as np
from McsPy import ureg

inf = float('inf')

def digitize(data, th_lo, th_hi):
    return np.where((data > th_lo) & (data < th_hi), 0, 1).astype('uint8')

def split_where(mask):
    mask = np.concatenate(([False], mask, [False] ))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    return idx.reshape((-1, 2))

def get_stream_data_in_range(stream, channels, t0=0, t1=inf):
    stream_data = {}

    for ch in channels:
        tick = stream.channel_infos[ch].sampling_tick
        tick = ureg.convert(tick.magnitude, tick.units, "second")

        idx_start = 0
        idx_end = stream.channel_data.shape[1]
        if t1 != inf:
            idx_end = int(t1 / tick)
        if t0 > 0:
            idx_start = int(t0 / tick)

        data, unit = stream.get_channel_in_range(ch, idx_start, idx_end)
        data = ureg.convert(data, unit, "microvolt")
        stream_data[ch] = data

    return stream_data

def get_timestamp_data_in_range(timestamps, channels, t0=0, t1=inf):
    timestamp_data = {}

    for ch in channels:
        if not ch in timestamps:
            continue
        ts, unit = timestamps[ch].get_timestamps()
        ts = ts.flatten()
        ts = np.array((ts * unit).to("second"))
        ts = np.ma.masked_outside(ts, t0, t1)
        ts = ts.compressed()
        timestamp_data[ch] = ts

    return timestamp_data

def spikes_to_bits(timestamps, stream, channels, t0=0, t1=inf):
    bits = []

    for ch in channels:
        ts, unit = timestamps[ch].get_timestamps()
        tick = stream.channel_infos[ch].sampling_tick
        # convert to ticks
        ts = (ts * unit).to(tick)/tick
        ts = ts.astype(int)

        idx_start = 0
        idx_end = stream.channel_data.shape[1]
        if t1 != inf:
            idx_end = int(ureg.convert(t1, "second", tick)/tick.magnitude)
        if t0 > 0:
            idx_start = int(ureg.convert(t0, "second", tick)/tick.magnitude)

        print(idx_start, idx_end)

        dd = np.zeros(stream.channel_data.shape[1])
        dd[ts] = 1
        dd = dd[idx_start:idx_end]
        bits.append(dd)

    return bits
