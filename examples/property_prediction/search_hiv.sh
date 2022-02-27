#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

for model in base large
do
    for norm in postln preln
    do
        for flag_m in 1 2 3 4
        do
            flag_step_size=0.001
            flag_mag=0
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "0,1" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.001
            flag_mag=0.001
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "2,3" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.001
            flag_mag=0.01
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "4,5" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.001
            flag_mag=0.1
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "6,7" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.01
            flag_mag=0
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "8,9" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.01
            flag_mag=0.001
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "10,11" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.01
            flag_mag=0.01
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "12,13" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
            flag_step_size=0.01
            flag_mag=0.1
            bash hiv_pre.sh $model $norm $flag_m $flag_step_size $flag_mag "14,15" 2>&1 > /blob/test_hiv/${model}_${norm}_${flag_m}_${flag_step_size}_${flag_mag}.log &
        done
    done
done
