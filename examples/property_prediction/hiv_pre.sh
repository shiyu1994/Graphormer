#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cleanup() {
    # kill all processes whose parent is this process
    echo "inner running clean up"
    pkill -9 -P $$
}

for sig in INT QUIT HUP TERM KILL; do
  trap "
    cleanup
    trap - $sig EXIT
    kill -s $sig "'"$$"' "$sig"
done
trap cleanup EXIT

n_gpu=$1
epoch=$2
max_epoch=$((epoch + 1))
batch_size=$3
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_percentage=$4
warmup_updates=$((tot_updates * warmup_percentage / 100))
lr=$5
flag_m=$6
flag_step_size=$7
flag_mag=$8
exp_dir=$9
base_or_large=${10}
postln_or_preln=${11}
gpu_id=${12}
root_path=${13}

for ckpt_id in 0 1 2 3 4 5
do
    save_dir=${exp_dir}/ng${n_gpu}_ep${epoch}_bs${batch_size}_wp${warmup_percentage}_l${lr}_fm${flag_m}_fss${flag_step_size}_fm${flag_mag}_bol${base_or_large}_pop${postln_or_preln}_gi${gpu_id}
    mkdir -p ${save_dir}
    mkdir -p ${save_dir}/${ckpt_id}
    result_path=${save_dir}/${ckpt_id}/result

    for seed in 0 1 2 3 4
    do
        model_path=${save_dir}/${ckpt_id}/${seed}
        log_path=${save_dir}/${ckpt_id}/${base_or_large}_${postln_or_preln}_${seed}
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

        if [[ ${postln_or_preln} == "preln" ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction_with_flag \
            --criterion binary_logloss_with_flag \
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
            --seed ${seed} \
            --flag-m ${flag_m} \
            --flag-step-size ${flag_step_size} \
            --flag-mag ${flag_mag} \
            --pre-layernorm >> ${log_path} 2>&1
        else
            CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction_with_flag \
            --criterion binary_logloss_with_flag \
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
            --seed ${seed} \
            --flag-m ${flag_m} \
            --flag-step-size ${flag_step_size} \
            --flag-mag ${flag_mag} >> ${log_path} 2>&1
        fi

        cd ../../graphormer/evaluate

        if [[ ${postln_or_preln} == "preln" ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate.py \
                --user-dir ../../graphormer \
                --num-workers 16 \
                --ddp-backend=legacy_ddp \
                --dataset-name ogbg-molhiv \
                --dataset-source ogb \
                --task graph_prediction \
                --arch graphormer_${base_or_large} \
                --num-classes 1 \
                --batch-size 64 \
                --save-dir ${model_path} \
                --split test \
                --metric auc \
                --seed ${seed} \
                --pre-layernorm >> ${log_path} 2>&1
            CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate.py \
                --user-dir ../../graphormer \
                --num-workers 16 \
                --ddp-backend=legacy_ddp \
                --dataset-name ogbg-molhiv \
                --dataset-source ogb \
                --task graph_prediction \
                --arch graphormer_${base_or_large} \
                --num-classes 1 \
                --batch-size 64 \
                --save-dir ${model_path} \
                --split valid \
                --metric auc \
                --seed ${seed} \
                --pre-layernorm >> ${log_path} 2>&1
        else
            CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate.py \
                --user-dir ../../graphormer \
                --num-workers 16 \
                --ddp-backend=legacy_ddp \
                --dataset-name ogbg-molhiv \
                --dataset-source ogb \
                --task graph_prediction \
                --arch graphormer_${base_or_large} \
                --num-classes 1 \
                --batch-size 64 \
                --save-dir ${model_path} \
                --split test \
                --metric auc \
                --seed ${seed} >> ${log_path} 2>&1
            CUDA_VISIBLE_DEVICES=${gpu_id} python -u evaluate.py \
                --user-dir ../../graphormer \
                --num-workers 16 \
                --ddp-backend=legacy_ddp \
                --dataset-name ogbg-molhiv \
                --dataset-source ogb \
                --task graph_prediction \
                --arch graphormer_${base_or_large} \
                --num-classes 1 \
                --batch-size 64 \
                --save-dir ${model_path} \
                --split valid \
                --metric auc \
                --seed ${seed} >> ${log_path} 2>&1
        fi
        rm -rf ${model_path}
        cd ../../examples/property_prediction/
    done
    python -u parse_results.py ${save_dir}/${ckpt_id} ${base_or_large}_${postln_or_preln} > ${result_path} 2>&1
done
