#!/bin/bash
for weights in $(cat weights.list)
do
  python create_runner.py -b 1024 "$weights" || \
    python create_runner.py -b 512 "$weights" || \
    python create_runner.py -b 64 "$weights" || \
    python create_runner.py -b 32 "$weights" || \
    continue
  python eval_runner.py "$weights"
done