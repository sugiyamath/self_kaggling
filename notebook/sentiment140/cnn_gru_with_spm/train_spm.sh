#!/bin/bash
python prepare_spm.py
spm_train --input spm_data.txt --model_prefix=m --vocab_size=8000
