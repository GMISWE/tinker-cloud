#!/usr/bin/env bash
# Cases for wait_rc of scripts/lib/cluster.sh. EX runs its command in a local shell, so no pod is
# needed. One line per case, `ok <case>` or `not ok <case>`; exit 1 if any case is not ok.
#
#   scripts/lib/cluster_wait_test.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/lib/cluster.sh
. "$here/cluster.sh"
EX() { bash -c "$1"; }
export WAIT_RC_POLL=1
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0
# run_named <name> <seconds>: a process whose command line holds <name>, alive for <seconds>
run_named() { printf 'sleep %s\n' "$2" > "$T/$1.sh"; bash "$T/$1.sh"; }
case_is() {  # <name> <want rc> <want stdout> <got rc> <got stdout>
  if [ "$2" = "$4" ] && [ "$3" = "$5" ]; then echo "ok $1"; else echo "not ok $1: want rc $2 '$3', got rc $4 '$5'"; fails=1; fi
}

# a run that ends and writes its status: the status is printed
run_named wait_rc_case_done 1; echo 7 > "$T/done.rc"
out=$(wait_rc "$T/done.rc" 'wait_rc_case_don[e]' 10 2>/dev/null); case_is "status written" 0 7 $? "$out"

# a run still alive when the wait starts, which then writes its status
( run_named wait_rc_case_live 3; echo 0 > "$T/live.rc" ) &
out=$(wait_rc "$T/live.rc" 'wait_rc_case_liv[e]' 20 2>/dev/null); case_is "status after a wait" 0 0 $? "$out"

# a run that was killed: no process and no file is rc 1, well before the limit
start=$SECONDS
out=$(wait_rc "$T/killed.rc" 'wait_rc_case_kille[d]' 600 2>/dev/null); rc=$?
[ $((SECONDS - start)) -lt 30 ] || rc="slow"
case_is "killed run" 1 "" "$rc" "$out"

# a run that outlives the limit is rc 2
run_named wait_rc_case_long 30 & long=$!
out=$(wait_rc "$T/long.rc" 'wait_rc_case_lon[g]' 3 2>/dev/null); case_is "limit reached" 2 "" $? "$out"
pkill -P $long; kill $long 2>/dev/null; wait $long 2>/dev/null
exit $fails
