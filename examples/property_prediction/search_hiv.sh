#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

flag_m=3
flag_step_size=0.001
flag_mag=0.001
num_gpu=$1
job_name=$2
exp_dir=/blob/search_hiv/${job_name}/

if [[ ${num_gpu} == "8" ]]; then
    for ckpt_id in 0 1 2 3 4 5 6 7 8 9 10
    do
        for seed in 0 1 2
        do
            bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" ${seed} ${exp_dir}/base_postln_${seed} ${ckpt_id} &
            bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" ${seed} ${exp_dir}/base_preln_${seed} ${ckpt_id} &
            bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" ${seed} ${exp_dir}/base_postln_${seed} ${ckpt_id} &
            bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" ${seed} ${exp_dir}/base_preln_${seed} ${ckpt_id} &
            wait
        done
    done
elif [[ ${num_gpu} == "16" ]]; then
    for seed in 0 1 2
    do
        bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" ${seed} ${exp_dir}/base_postln_${seed} &
        bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" ${seed} ${exp_dir}/base_preln_${seed} &
        bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" ${seed} ${exp_dir}/base_postln_${seed} &
        bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" ${seed} ${exp_dir}/base_preln_${seed} &
        bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "8,9" ${seed} ${exp_dir}/base_postln_${seed} &
        bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "10,11" ${seed} ${exp_dir}/base_preln_${seed} &
        bash hiv_pre.sh base postln ${flag_m} ${flag_step_size} ${flag_mag} "12,13" ${seed} ${exp_dir}/base_postln_${seed} &
        bash hiv_pre.sh base preln ${flag_m} ${flag_step_size} ${flag_mag} "14,15" ${seed} ${exp_dir}/base_preln_${seed} &
        wait
    done
fi
