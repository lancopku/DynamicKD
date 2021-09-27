GPU=2
TEACHER_MODEL_PATH=/path/to/trained_teacher
SL=6 # student num layer
EPOCH=3
LR=2e-5
for STRGY in 'margin'   # data selection ['margin', 'lc', 'entropy', 'none']
do
for RATIO in 0.5 # selection ratio r
do
for TASK_NAME in mrpc 
do
for SEED in $(seq 1 3) #1234
do
OUTPUT_DIR=data_selection_ckpts/${TASK_NAME}-sl${SL}-msekd-seed${SEED}-epoch${EPOCH}-LR${LR}-STRGY${STRGY}-sr${RATIO}

CUDA_VISIBLE_DEVICES=$GPU python dynamic_data.py --selection_strategy $STRGY --selection_ratio $RATIO \
  --model_name_or_path bert-base-uncased  \
  --teacher $TEACHER_MODEL_PATH \
  --student_num_layers $SL  --warmup_ratio 0.1  \
  --kd_alpha 1.0 \
  --ce_alpha 1.0 \
  --seed $SEED \
  --task_name $TASK_NAME \
  --fp16  \
  --do_eval --do_train  \
  --max_seq_length 128 \
  --per_device_train_batch_size 32  \
  --per_device_eval_batch_size 64  --overwrite_output_dir \
  --save_total_limit 1 \
  --logging_steps 200  \
  --evaluation_strategy steps \
  --learning_rate $LR \
  --num_train_epochs $EPOCH  \
  --output_dir $OUTPUT_DIR
done
done
done
done
