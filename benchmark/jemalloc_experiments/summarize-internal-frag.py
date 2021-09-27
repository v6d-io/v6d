#!/usr/bin/env python3

# See summarize.py for the input requirement and how to run the script.

import json
import sys

def main(contents):
    bins = contents['jemalloc']['arenas']['bin']
    lextents = contents['jemalloc']['arenas']['lextent']

    binstats = contents['jemalloc']['stats.arenas']['merged']['bins']
    lextentstats = contents['jemalloc']['stats.arenas']['merged']['lextents']

    assert(len(bins) == len(binstats))
    assert(len(lextents) == len(lextentstats))

    nbins = len(bins)
    nsizes = nbins + len(lextents)
    bin_live_c_sum = 0

    print( "%4s"  "%14s"  "%14s"
           "%16s"            "%12s"        "%12s"       "%18s"
           "%16s"             "%12s"         "%12s"        "%18s" %
          ("ind", "size", "total",
           "live_requested", "live_count", "live_frag", "total_live_frag",
           "accum_requested", "accum_count", "accum_frag", "total_accum_frag"))

    for ind in range(nsizes):
        if ind < nbins:
            meta = bins[ind]
            stats = binstats[ind]
            count = stats['curregs']
        else:
            ilextent = ind - nbins
            meta = lextents[ilextent]
            stats = lextentstats[ilextent]
            count = stats['curlextents']

        live_r = stats['prof_live_requested']
        live_c = stats['prof_live_count']
        accum_r = stats['prof_accum_requested']
        accum_c = stats['prof_accum_count']

        if ind < nbins:
            count += live_c
            bin_live_c_sum += live_c
        elif ind == nbins:
            count -= bin_live_c_sum

        if not count:
            continue

        size = meta['size']
        total = count * size

        live_frag = 0.0 if live_c == 0 else 1.0 - live_r / (live_c * size)
        total_live_frag = int(total * live_frag)

        accum_frag = 0.0 if accum_c == 0 else 1.0 - accum_r / (accum_c * size)
        total_accum_frag = int(total * accum_frag)

        print( "%4d" "%14d" "%14d" "%16d"  "%12d"  "%12.4f"   "%18d"
               "%16d"   "%12d"   "%12.4f"    "%18d" %
              (ind,  size,  total, live_r, live_c, live_frag, total_live_frag,
               accum_r, accum_c, accum_frag, total_accum_frag))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "my_stats.json")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        contents = json.load(f)
    main(contents)
