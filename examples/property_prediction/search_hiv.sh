#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

flag_m=3
flag_step_size=0.001
flag_mag=0.001

bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" 0 large_postln_0 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" 0 large_preln_0 &
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" 1 large_postln_1 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" 1 large_preln_1 &
wait
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" 2 large_postln_2 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" 2 large_preln_2 &
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" 3 large_postln_3 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" 3 large_preln_3 &
wait
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" 4 large_postln_4 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" 4 large_preln_4 &
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" 5 large_postln_5 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" 5 large_preln_5 &
wait
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" 6 large_postln_6 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" 6 large_preln_6 &
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" 7 large_postln_7 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" 7 large_preln_7 &
wait
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "0,1" 8 large_postln_8 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "2,3" 8 large_preln_8 &
bash hiv_pre.sh large postln ${flag_m} ${flag_step_size} ${flag_mag} "4,5" 9 large_postln_9 &
bash hiv_pre.sh large preln ${flag_m} ${flag_step_size} ${flag_mag} "6,7" 9 large_preln_9 &
wait
