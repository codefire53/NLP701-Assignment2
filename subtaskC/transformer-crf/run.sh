#!/bin/bash
exp_name="exp_1"
seed_value=42
python training.py \
  --model_path "SpanBERT/spanbert-base-cased" \
  --model_checkpoint_dir "./runs/$exp_name/SpanBERT-spanbert-base-cased" \
  --train_file "../data/subtaskC_train-train.jsonl" \
  --load_best_model_at_end True \
  --dev_file "../data/subtaskC_train-dev.jsonl" \
  --test_files ../data/subtaskC_dev.jsonl \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --do_train False \
  --do_predict True \
  --seed $seed_value \
  --output_dir "./runs/$exp_name" \
  --logging_dir "./runs/$exp_name/logs" \
  --num_train_epochs 30 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --auto_find_batch_size True \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --max_length 512
