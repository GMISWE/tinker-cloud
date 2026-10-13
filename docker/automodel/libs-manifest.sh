#!/bin/sh
# Print "<path> <sha256 of resolved file>" for every shared library at depth 1
# of the system library dirs, sorted. Dangling links are skipped.
set -eu
find /usr/local/lib /usr/lib/x86_64-linux-gnu -maxdepth 1 \( -type f -o -type l \) -name '*.so*' \
  | LC_ALL=C sort -u \
  | while read -r p; do
      r=$(readlink -f "$p")
      if [ -f "$r" ]; then echo "$p $(sha256sum "$r" | cut -c1-64)"; fi
    done \
  | LC_ALL=C sort
