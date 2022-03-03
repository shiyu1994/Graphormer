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
num_workers_per_gpu=$9
num_gpu_per_process=${10}

gpu_id=0

if [[ "${num_gpu_per_process}" != "1" && "${num_gpu_per_process}" != "2" && "${num_gpu_per_process}" != "4" && "${num_gpu_per_process}" != "8" && "${num_gpu_per_process}" != "16" ]]; then
    echo "Unsupported num_gpu_per_process ${num_gpu_per_process}"
    exit
fi

echo "============================= Running a simple task to download MolHIV data ============================="
#bash get_hiv_data.sh
echo "============================= Finished ============================="

for lr in 2e-4 8e-4 4e-5 8e-5
do
    for flag_m in 1 2 3 4 6
    do
        for flag_step_size in 0.001 0.01 0.1 0.2 0.0001
        do
            for flag_mag in 0.001 0.01 0.1 0 0.0001
            do
                gpu_ids="${gpu_id}"
                for (( c=1; c<num_gpu_per_process; c++ ))
                do
                    gpu_ids+=",$((gpu_id + c))"
                done
                bash hiv_pre.sh ${num_gpu_per_process} ${epoch} ${batch_size} ${warmup_percentage} ${lr} ${flag_m} ${flag_step_size} ${flag_mag} ${exp_dir} ${base_or_large} ${postln_or_preln} ${gpu_ids} ${root} ${num_workers_per_gpu} &
                echo "Dispatching to GPU(s) ${gpu_ids}"
                gpu_id=$((gpu_id+num_gpu_per_process))
                if [[ "${gpu_id}" == "${num_gpu}" || "${gpu_id}" == "16" ]]; then
                    echo "Waiting"
                    wait
                    gpu_id=0
                fi
            done
        done
    done
done
