#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

num_gpu=$1
job_name=$2
exp_dir=/blob/search_hiv/${job_name}/
base_or_large=$3
postln_or_preln=$4

gpu_id=0
for epoch in 8 16
do
    for batch_size in 128 256 512 64
    do
        for warmup_percentage in 6 10 20 3
        do
            for lr in 2e-4 8e-4 4e-5 8e-5
            do
                for flag_m in 1 2 3 4 6
                do
                    for flag_step_size in 0.001 0.01 0.1 0.2 0.0001
                    do
                        for flag_mag in 0.001 0.01 0.1 0 0.0001
                        do
                            bash 1 ${epoch} ${batch_size} ${warmup_percentage} ${lr} ${flag_m} ${flag_step_size} ${flag_mag} ${exp_dir} ${base_or_large} ${postln_or_preln} ${gpu_id} &
                            gpu_id=$((gpu_id+1))
                            if [[ gpu_id == num_gpu ]]; then
                                wait
                                gpu_id=0
                            fi
                        done
                    done
                done
            done
        done
    done
done
