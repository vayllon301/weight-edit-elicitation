#!/bin/zsh
# M2 queues. Each step is restartable: finished runs are skipped. Usage: queue.sh L | WH | SFT
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
case $1 in
  L)  orgs=(L) ;;
  WH) orgs=(W H) ;;
  SFT)
    for o in L W H; do for n in 1 16; do
      $PY -I baselines/sft.py --org $o --n $n || exit 1
    done; done
    exit 0 ;;
esac
for o in $orgs; do
  if [[ $o != L ]]; then
    $PY -I baselines/noise_select.py --org $o || exit 1
    $PY -I baselines/run_edits.py --org $o --family random || exit 1
  fi
  $PY -I baselines/few_shot.py --org $o || exit 1
  for f in proj add exp8; do
    $PY -I baselines/run_edits.py --org $o --family $f || exit 1
  done
done
