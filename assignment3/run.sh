#!/bin/bash

if [[ $# == 1 ]]; then
    label="${1}_"
fi

now=$(date +"%y%m%d_%H%M%S")
datafile=/data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat

export SLURM_JOB_CPUS_PER_NODE=32
sbatch --cpus-per-task=32 --output output/${label}${now}.out /opt/local/bin/run_job.sh \
        plot_problem1.py --output output/${label}${now}.png ${datafile}
