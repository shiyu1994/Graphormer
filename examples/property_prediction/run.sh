#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ulimit -c unlimited

if [[ $2 == "preln" ]]; then
    fairseq-train \
    --user-dir ../../graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name $1 \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_$3 \
    --num-classes 1 \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
    --lr 2e-4 --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --save-dir ./ckpts \
    --pre-layernorm
elif [[ $2 == "sandwich" ]]; then
    fairseq-train \
    --user-dir ../../graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name $1 \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_$3 \
    --num-classes 1 \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
    --lr 2e-4 --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --save-dir ./ckpts \
    --sandwich-layernorm
elif [[ $2 == "postln" ]]; then
    fairseq-train \
    --user-dir ../../graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name $1 \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_$3 \
    --num-classes 1 \
    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
    --lr 2e-4 --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --save-dir ./ckpts
fi
