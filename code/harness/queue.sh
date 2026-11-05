#!/bin/zsh
# The runs one after another (one GPU), within one background command's limit. Call it again until it says done: each
# driver resumes from state.json.
# the design (04-11-2026): one L-neutral run first, then stop and review measured hours and tokens. So by default this runs
# only L-neutral-1. After the review, QUEUE_REVIEWED=1 runs all six; the first is already done and returns at once.
# usage: code/harness/queue.sh [seconds, default 6600]
# Exit: 0 all done; 2 call again (wall limit, or waiting for the model); 3 a run halted and needs a human; 1 a driver crashed;
# 4 another driver holds the run. Anything but 0 and 2 is for a human; either way queue-status.txt in the runs folder says what
# happened and how to go on, with the run's own status.txt inside.
# QUEUE_RUNS replaces the run list: a dry run against a stub, or holding later runs back until a review is done.
# QUEUE_ARGS adds driver arguments: only for a dry run.
cd ${0:A:h}/../..
budget=${1:-6600}
runs_dir=${HARNESS_RUNS:-code/runs}
status_file=$runs_dir/queue-status.txt
kill_retries=${QUEUE_KILL_RETRIES:-40}  # with the wait, 20 min; the last event of this kind ran 14 min 41 s
kill_wait=${QUEUE_KILL_WAIT:-30}
if [[ -n ${QUEUE_RUNS:-} ]]; then
  runs=(${=QUEUE_RUNS})
elif [[ ${QUEUE_REVIEWED:-} == 1 ]]; then
  runs=(L-neutral-1 L-auditor-1 W-auditor-1 L-neutral-2 Hft-auditor-1 L-auditor-2)
else
  runs=(L-neutral-1)
fi
go_on_prefix=
[[ -n ${QUEUE_RUNS:-} ]] && go_on_prefix="QUEUE_RUNS='$QUEUE_RUNS' "  # the resume line must not widen the run list
[[ ${QUEUE_REVIEWED:-} == 1 ]] && go_on_prefix+='QUEUE_REVIEWED=1 '

note() {  # note <exit> <run> <meaning>
  mkdir -p $runs_dir
  {
    echo "queue exit: $1"
    echo "meaning: $3"
    echo "run: $2"
    echo "at: $(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "go on with: ${go_on_prefix}code/harness/queue.sh $budget"
    for d in $runs_dir/agent-$2 $runs_dir/harness-scratch-$2; do
      if [[ -f $d/status.txt ]]; then
        echo "--- $d/status.txt"
        cat $d/status.txt
      fi
    done
  } > $status_file
}

start=$(date +%s)
for run in $runs; do
  left=$(( budget - ($(date +%s) - start) ))
  if (( left <= 0 )); then
    note 2 $run "wall limit between runs: call again"
    echo "wall limit: call again"
    exit 2
  fi
  kills=0
  while true; do
    code/.venv/bin/python -I code/harness/driver.py --run $run --max-wall-seconds $left ${=QUEUE_ARGS:-}
    rc=$?
    # 137 is SIGKILL: macOS killing the interpreter at exec, or a hard limit. state.json makes resuming safe
    (( rc == 137 && kills < kill_retries )) || break
    kills=$(( kills + 1 ))
    echo "$run: driver killed ($kills of $kill_retries), trying again in $kill_wait s"
    sleep $kill_wait
    left=$(( budget - ($(date +%s) - start) ))
    if (( left <= 0 )); then
      rc=2
      break
    fi
  done
  (( rc == 0 )) && continue
  case $rc in
    2) meaning="stopped, resumable: call again" ;;
    3) meaning="HALTED: a human must look before it goes on" ;;
    4) meaning="another driver holds this run" ;;
    137) meaning="the driver was killed $kills times in a row: check the machine" ;;
    *) meaning="the driver crashed (exit $rc)" ;;
  esac
  note $rc $run "$meaning"
  echo "$run stopped with exit $rc: $meaning"
  echo "see $status_file"
  exit $rc
done
note 0 "$runs" "all done"
echo "all done: $runs"
