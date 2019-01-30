#!/bin/bash
./download.sh
./train_spm.sh
python train.py > result.txt
