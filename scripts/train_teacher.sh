GPU=0
TASK_NAME=rte
MODEL="bert-base" # bert-large
CUDA_VISIBLE_DEVICtES=$GPU python run_glue.py \
  --model_name_or_path "$MODEL-uncased" \
  --task_name $TASK_NAME \
  --do_train --fp16  \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size  32 \
  --per_device_eval_batch_size 64\
  --save_total_limit 1 \
  --logging_steps 500 \
  --evaluation_strategy steps --warmup_ratio 0.05  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ckpts/${TASK_NAME}-$MODEL

