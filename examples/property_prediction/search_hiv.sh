#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cleanup() {
    # kill all processes whose parent is this process
    echo "runnning clean up"
    pkill -9 -P $$
}

for sig in INT QUIT HUP TERM KILL; do
  trap "
    cleanup
    trap - $sig EXIT
    kill -s $sig "'"$$"' "$sig"
done
trap cleanup EXIT

num_gpu=$1
job_name=$2
base_or_large=$3
postln_or_preln=$4
root=$5
exp_dir=${root}/${job_name}/
epoch=$6
batch_size=$7
warmup_percentage=$8

gpu_id=0

for lr in 2e-4 8e-4 4e-5 8e-5
do
    for flag_m in 1 2 3 4 6
    do
        for flag_step_size in 0.001 0.01 0.1 0.2 0.0001
        do
            for flag_mag in 0.001 0.01 0.1 0 0.0001
            do
                bash hiv_pre.sh 1 ${epoch} ${batch_size} ${warmup_percentage} ${lr} ${flag_m} ${flag_step_size} ${flag_mag} ${exp_dir} ${base_or_large} ${postln_or_preln} ${gpu_id} ${root} &
                gpu_id=$((gpu_id+1))
                echo "Dispatching to GPU ${gpu_id}"
                if [[ "${gpu_id}" == "${num_gpu}" || "${gpu_id}" == "16" ]]; then
                    echo "Waiting"
                    wait
                    gpu_id=0
                fi
            done
        done
    done
done
