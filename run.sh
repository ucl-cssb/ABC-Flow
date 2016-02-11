#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code
#
python run-abc-flow.py -i model-gardner/input_file.xml -o gard_2D >& log.gard2D