#!/bin/bash
# Script để chạy training với virtual environment
cd /home/huynguyen/source/Trash-Detection/training-model
source training_env/bin/activate
python main.py "$@"