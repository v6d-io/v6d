#!/usr/bin/env python3

# this file is borrowed and modified from https://stackoverflow.com/a/65635592/5080177
#

import os
import sys

from datetime import datetime
from tarfile import open
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

compression = ZIP_DEFLATED

from_name = sys.argv[1]
if len(sys.argv) > 2:
    to_name = sys.argv[2]
else:
    to_name = os.path.splitext(os.path.splitext(from_name)[0])[0] + '.zip'

with open(name=from_name, mode='r|gz') as tarf:
    with ZipFile(file=to_name, mode='w', compression=compression) as zipf:
        for m in tarf:
            mtime = datetime.fromtimestamp(m.mtime)
            zinfo = ZipInfo(
                filename=m.name,
                date_time=(mtime.year, mtime.month, mtime.day, mtime.hour, mtime.minute, mtime.second)
            )
            if not m.isfile():
                # for directories and other types
                continue
            f = tarf.extractfile(m)
            fl = f.read()
            zipf.writestr(zinfo, fl, compress_type=compression)

print('converted %s to %s.' % (from_name, to_name))
