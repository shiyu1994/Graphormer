#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

flag_m=3
flag_step_size=0.001
flag_mag=0.001

for model in base large
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        bash hiv_pre.sh ${model} postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" ${seed} 2>&1 > ${model}_postln_${seed} &
        bash hiv_pre.sh ${model} preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" ${seed} 2>&1 > ${model}_preln_${seed} &
        wait
    done
done
