#!/usr/bin/env bash

python ~/tools/transformers/examples/run_glue.py \
       --model_type roberta \
       --model_name_or_path roberta-base \
       --task_name IRC \
       --do_train \
       --do_eval \
       --do_lower_case \
       --data_dir ~/data/irc-disentanglement/data/pairs_corpus/data_50  \
       --max_seq_length 128 \
       --per_gpu_eval_batch_size=32 \
       --per_gpu_train_batch_size=32 \
       --learning_rate 2e-5 \
       --num_train_epochs 20.0 \
       --output_dir /data/sebastien/ascii_roberta_irc_50_full \
       --save_steps 1000 \
       --eval_all_checkpoints \
       --evaluate_during_training \
       --logging_steps 1000 \
       --overwrite_output_dir
