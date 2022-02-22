#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ulimit -c unlimited

git_commit=$(git rev-parse HEAD)
echo "Using git commit ${git_commit}"
checkpoint_dir=checkpoint_$1_$2_$3_$4_$5_$6_${7}_${git_commit}
mkdir /blob/mol/${checkpoint_dir}

if [[ $4 == "fp16" ]]; then
    if [[ $3 == "graphormer_v2" ]]; then
        if [[ $5 == "preln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --pre-layernorm
        elif [[ $5 == "sandwich" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --sandwich-layernorm
        elif [[ $5 == "postln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir}
        fi
    elif [[ $3 == "graphormer_v1" ]]; then
        if [[ $5 == "preln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --pre-layernorm
        elif [[ $5 == "sandwich" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --sandwich-layernorm
        elif [[ $5 == "postln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --fp16 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir}
        fi
    fi
else
    if [[ $3 == "graphormer_v2" ]]; then
        if [[ $5 == "preln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --pre-layernorm
        elif [[ $5 == "sandwich" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --sandwich-layernorm
        elif [[ $5 == "postln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir}
        fi
    elif [[ $3 == "graphormer_v1" ]]; then
        if [[ $5 == "preln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --pre-layernorm
        elif [[ $5 == "sandwich" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir} \
            --sandwich-layernorm
        elif [[ $5 == "postln" ]]; then
            fairseq-train \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name $2 \
            --dataset-source ogb \
            --task graph_prediction \
            --criterion l1_loss \
            --arch graphormer_$1_v1 \
            --num-classes 1 \
            --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
            --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
            --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
            --lr ${7} --end-learning-rate 1e-9 \
            --batch-size $6 \
            --data-buffer-size 20 \
            --max-epoch 300 \
            --save-dir /blob/mol/${checkpoint_dir}
        fi
    fi
fi
