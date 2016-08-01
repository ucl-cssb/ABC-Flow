#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/cuda-sim-code
#
python run-abc-flow.py -i model-gardner/input_file_2D.xml -o gard_2D >& log.gard2d


