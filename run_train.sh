#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --dataset multi_all --source1 real --source2 painting --target sketch --net $3 --save_check
