GPU=2
SMALL_TEACHER=ckpts/rte-bert-base # /path/to/small_teacher
LARGE_TEACHER=ckpts/rte-bert-large # /path/to/large_teacher
SL=6 #student num layer
EPOCH=3
LR=2e-5
for STA in 1.0  # kd loss weight for small teacher model
do
for LTA in 1.0 # kd loss weight for large teacher model
do
for STRGY in 'hard' 'soft'
do
for TASK_NAME in rte
do
for SEED in 1 #$(seq 1 3) #1234
do
CUDA_VISIBLE_DEVICES=$GPU python dynamic_teacher.py \
  --model_name_or_path "bert-base-uncased" \
  --small_teacher $SMALL_TEACHER \
  --large_teacher $LARGE_TEACHER \
  --small_teacher_alpha $STA \
  --large_teacher_alpha $LTA \
  --student_num_layers $SL  --warmup_ratio 0.05 \
  --kd_alpha 1.0  \
  --ce_alpha 1.0 \
  --seed $SEED \
  --task_name $TASK_NAME \
  --do_train --fp16  \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 --uncertainty_mode $STRGY \
  --per_device_eval_batch_size 64  --overwrite_output_dir \
  --save_total_limit 1 \
  --logging_steps 20   \
  --evaluation_strategy steps \
  --learning_rate $LR \
  --num_train_epochs $EPOCH  \
  --output_dir ${STRGY}_selection_ckpts/${TASK_NAME}-sl${SL}-sta${STA}-lta${LTA}-msekd-seed${SEED}-epoch${EPOCH}-LR${LR}

done
done
done
done
done
