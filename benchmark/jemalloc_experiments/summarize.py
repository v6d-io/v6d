#!/usr/bin/env python3
#
# First, export stats to a file with at least the options JM.  We only need
# merged arena stats, so if you expose stats output on some webserver status
# path, the right command is something like:
#
# wget -O my_stats.json server:port/pprof/mallocstats?opts=JMa
# summarize.py my_stats.json

import json
import sys

def human_size_str(size):
    strings = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]
    ind = 0
    while size >= 1024 and ind < len(strings):
        ind += 1
        size /= 1024
    return "{:0.1f} {}".format(size, strings[ind])

def avg_size_str(nbytes, n):
    if n == 0:
        return "n/a"
    return human_size_str(nbytes / n)

def waste_in_bin(binstat, bin):
    # *Live* objects
    curregs = binstat['curregs']
    nslabs = binstat['curslabs']
    # *all objects*
    nregs = nslabs * bin['nregs']
    waste = (nregs - curregs) * bin['size']
    return waste

def live_in_bin(binstat, bin):
    # *Live* objects
    curregs = binstat['curregs']
    live = curregs * bin['size']
    return live

def main(contents):
    # Bins
    print("-----------------------------------------------------")
    print("BINS")
    print("-----------------------------------------------------")
    bins = contents['jemalloc']['arenas']['bin']
    binstats = contents['jemalloc']['stats.arenas']['merged']['bins']
    sum_bin_waste = 0
    sum_bin_live = 0
    for i in range(len(binstats)):
        binstat = binstats[i]
        bin = bins[i]
        waste = waste_in_bin(binstat, bin)
        live = live_in_bin(binstat, bin)
        print("In bin {} ({} bytes), wasted memory is {}, live is {}".format(i,
            bin['size'], human_size_str(waste), human_size_str(live)))
        sum_bin_waste += waste
        sum_bin_live += live

    # Extents
    print("-----------------------------------------------------")
    print("EXTENTS")
    print("-----------------------------------------------------")
    sum_extent_waste = 0
    extents = contents['jemalloc']['stats.arenas']['merged']['extents']
    sum_allocated = 0
    sum_dirty = 0
    sum_muzzy = 0
    for i, extent in enumerate(extents):
        dirty = extent['dirty_bytes']
        ndirty = extent['ndirty']
        muzzy = extent['muzzy_bytes']
        nmuzzy = extent['nmuzzy']
        sum_dirty += dirty
        sum_muzzy += muzzy
        if ndirty == 0 and nmuzzy == 0:
            continue
        print("In pszind {}, dirty: {}, muzzy: {}. avgdirty: {}. avgmuzzy: {}"
                .format(i, human_size_str(dirty), human_size_str(muzzy),
                    avg_size_str(dirty, ndirty), avg_size_str(muzzy, nmuzzy)))
    # One-offs
    tcache_bytes = contents['jemalloc']['stats.arenas']['merged']['tcache_bytes']
    metadata_bytes = contents['jemalloc']['stats']['metadata']
    user_bytes = contents['jemalloc']['stats']['allocated']

    print("-----------------------------------------------------")
    print("TOTALS")
    print("-----------------------------------------------------")
    print("Total waste across bins: {}".format(human_size_str(sum_bin_waste)))
    print("Total waste across extents: {} dirty, {} muzzy".format(
        human_size_str(sum_dirty), human_size_str(sum_muzzy)))
    print("Total waste in tcaches: {}".format(human_size_str(tcache_bytes)))
    print("Total waste in metadata: {}".format(human_size_str(metadata_bytes)))
    print("Total user bin bytes: {}".format(human_size_str(sum_bin_live)))
    print("Total user bytes: {}".format(human_size_str(user_bytes)))
    print("Waste / user bytes ratio: {:0.3f}".format((sum_bin_waste
        + sum_dirty + tcache_bytes + metadata_bytes)/user_bytes))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "my_stats.json")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        contents = json.load(f)
    main(contents)
