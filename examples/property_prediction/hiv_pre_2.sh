#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=2
epoch=8
max_epoch=$((epoch + 1))
batch_size=64
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates/10))

if [[ $1 == "large" ]]; then
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=2,3 fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction_with_flag \
        --criterion binary_logloss_with_flag \
        --arch graphormer_large \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
        --lr 8e-5 --end-learning-rate 1e-9 \
        --batch-size $batch_size \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 24 \
        --encoder-embed-dim 1024 \
        --encoder-ffn-embed-dim 1024 \
        --encoder-attention-heads 32 \
        --max-epoch $max_epoch \
        --save-dir ./ckpts_$1_$2_$3_$4_$5 \
        --pretrained-model-name test_ckpts/checkpoint_$1_$2.pt \
        --seed 1 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 \
        --pre-layernorm
    else
        CUDA_VISIBLE_DEVICES=2,3 fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction_with_flag \
        --criterion binary_logloss_with_flag \
        --arch graphormer_large \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size $batch_size \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 24 \
        --encoder-embed-dim 1024 \
        --encoder-ffn-embed-dim 1024 \
        --encoder-attention-heads 32 \
        --max-epoch $max_epoch \
        --save-dir ./ckpts_$1_$2_$3_$4_$5 \
        --pretrained-model-name test_ckpts/checkpoint_$1_$2.pt \
        --seed 1 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5
    fi
else
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=2,3 fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction_with_flag \
        --criterion binary_logloss_with_flag \
        --arch graphormer_base \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size $batch_size \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 12 \
        --encoder-embed-dim 768 \
        --encoder-ffn-embed-dim 768 \
        --encoder-attention-heads 32 \
        --max-epoch $max_epoch \
        --save-dir ./ckpts_$1_$2_$3_$4_$5 \
        --pretrained-model-name test_ckpts/checkpoint_$1_$2.pt \
        --seed 1 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 \
        --pre-layernorm
    else
        CUDA_VISIBLE_DEVICES=2,3 fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction_with_flag \
        --criterion binary_logloss_with_flag \
        --arch graphormer_base \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size $batch_size \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 12 \
        --encoder-embed-dim 768 \
        --encoder-ffn-embed-dim 768 \
        --encoder-attention-heads 32 \
        --max-epoch $max_epoch \
        --save-dir ./ckpts_$1_$2_$3_$4_$5 \
        --pretrained-model-name test_ckpts/checkpoint_$1_$2.pt \
        --seed 1 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5
    fi
fi
