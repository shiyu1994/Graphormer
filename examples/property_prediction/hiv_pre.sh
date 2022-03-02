#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=2
epoch=8
max_epoch=$((epoch + 1))
batch_size=64
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates * 6 / 100))

save_dir=/mnt/shiyu/ckpts_$1_$2_$3_$4_$5_$7
pretrained_model_name=/blob/search_hiv/test_ckpts/checkpoint_$1_$2_$9.pt

if [[ $1 == "large" ]]; then
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=$6 fairseq-train \
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
        --save-dir ${save_dir} \
        --pretrained-model-name ${pretrained_model_name} \
        --seed $7 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 \
        --pre-layernorm > $8 2>&1
    else
        CUDA_VISIBLE_DEVICES=$6 fairseq-train \
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
        --save-dir ${save_dir} \
        --pretrained-model-name ${pretrained_model_name} \
        --seed $7 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 > $8 2>&1
    fi
else
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=$6 fairseq-train \
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
        --save-dir ${save_dir} \
        --pretrained-model-name ${pretrained_model_name} \
        --seed $7 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 \
        --pre-layernorm > $8 2>&1
    else
        CUDA_VISIBLE_DEVICES=$6 fairseq-train \
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
        --save-dir ${save_dir} \
        --pretrained-model-name pcqm4mv1_graphormer_base \
        --seed $7 \
        --flag-m $3 \
        --flag-step-size $4 \
        --flag-mag $5 > $8 2>&1
    fi
fi

cd ../../graphormer/evaluate
if [[ $1 == "large" ]]; then
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_large \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split test \
            --metric auc \
            --seed $7 \
            --pre-layernorm >> $8 2>&1
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_large \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split valid \
            --metric auc \
            --seed $7 \
            --pre-layernorm >> $8 2>&1
    else
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_large \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split test \
            --metric auc \
            --seed $7 >> $8 2>&1
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_large \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split valid \
            --metric auc \
            --seed $7 >> $8 2>&1
    fi
else
    if [[ $2 == "preln" ]]; then
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_base \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split test \
            --metric auc \
            --seed $7 \
            --pre-layernorm >> $8 2>&1
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_base \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split valid \
            --metric auc \
            --seed $7 \
            --pre-layernorm >> $8 2>&1
    else
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_base \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split test \
            --metric auc \
            --seed $7 >> $8 2>&1
        CUDA_VISIBLE_DEVICES=$6 python -u evaluate.py \
            --user-dir ../../graphormer \
            --num-workers 16 \
            --ddp-backend=legacy_ddp \
            --dataset-name ogbg-molhiv \
            --dataset-source ogb \
            --task graph_prediction \
            --arch graphormer_base \
            --num-classes 1 \
            --batch-size 64 \
            --save-dir ${save_dir} \
            --split valid \
            --metric auc \
            --seed $7 >> $8 2>&1
    fi
fi
