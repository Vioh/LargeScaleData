#!/bin/bash

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: ${0} script_name [label]"
    exit 0
fi

if [[ $# == 2 ]]; then
    label="${2}_"
fi

now=$(date +"%y%m%d_%H%M%S")
data_file=/data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
script_name=${1}

export SLURM_JOB_CPUS_PER_NODE=32
export PYTHONUNBUFFERED=TRUE

sbatch --cpus-per-task=32 --output output/${label}${now}.out /opt/local/bin/run_job.sh \
        plot.py -o output/${label}${now}.png -s ${script_name} ${data_file}
