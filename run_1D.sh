#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code

#
python run-abc-flow.py -i model-gardner/input_file_1D.xml -o gard_1D >& log.gard1D


