#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=1
epoch=8
max_epoch=$((epoch + 1))
batch_size=128
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_percentage=6
warmup_updates=$((tot_updates * warmup_percentage / 100))
lr=2e-4
flag_m=3
flag_step_size=0.001
flag_mag=0.001
exp_dir=/home/shiyu/Graphormer-dev/Graphormer/examples/property_prediction/try/
base_or_large=base
postln_or_preln=postln
gpu_id=${1}
root_path=/home/shiyu/Graphormer-dev/Graphormer/examples/property_prediction/
num_workers_per_gpu=8
num_workers=$((n_gpu * num_workers_per_gpu))
seed=${2}
ckpt_id=0

save_dir=${exp_dir}/${seed}
mkdir -p ${save_dir}
model_path=${save_dir}
log_path=${save_dir}/${base_or_large}_${postln_or_preln}_${seed}
mkdir -p ${model_path}
touch ${log_path}
pretrained_model_name=${root_path}/test_ckpts/checkpoint_${base_or_large}_${postln_or_preln}_${ckpt_id}.pt

echo "======================================== begin hyper parameters ========================================" >> ${log_path}
echo "n_gpu=${n_gpu}" >> ${log_path}
echo "epoch=${epoch}" >> ${log_path}
echo "batch_size=${batch_size}" >> ${log_path}
echo "warmup_percentage=${warmup_percentage}" >> ${log_path}
echo "lr=${lr}" >> ${log_path}
echo "flag_m=${flag_m}" >> ${log_path}
echo "flag_step_size=${flag_step_size}" >> ${log_path}
echo "flag_mag=${flag_mag}" >> ${log_path}
echo "exp_dir=${exp_dir}" >> ${log_path}
echo "base_or_large=${base_or_large}" >> ${log_path}
echo "postln_or_preln=${postln_or_preln}" >> ${log_path}
echo "ckpt_id=${ckpt_id}" >> ${log_path}
echo "gpu_id=${gpu_id}" >> ${log_path}
echo "seed=${seed}" >> ${log_path}
echo "======================================== end hyper parameters ========================================" >> ${log_path}

CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-train \
    --user-dir ../../graphormer \
    --num-workers ${num_workers} \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molhiv \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion binary_logloss \
    --arch graphormer_${base_or_large} \
    --num-classes 1 \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
    --lr ${lr} --end-learning-rate 1e-9 \
    --batch-size $batch_size \
    --fp16 \
    --data-buffer-size 20 \
    --max-epoch $max_epoch \
    --save-dir ${model_path} \
    --pretrained-model-name ${pretrained_model_name} \
    --seed ${gpu_id} >> ${log_path} 2>&1
