GPU=0
TEACHER_MODEL_PATH=/path/to/trained_teacher
SL=6 # student layer number
EPOCH=3
LR=2e-5
KLa=1.0
KL_ALPHA=1.0
CE_ALPHA=1.0
for RALPHA in 1 # rte # 4
do
for STRGY in 'uncertainty' # none
do
for TASK_NAME in rte #
do
for SEED in 1 # 2 3  #
do
OUTPUT_DIR=./${TASK_NAME}-sl${SL}-msekd-seed${SEED}-epoch${EPOCH}-LR${LR}-KLALPHA${KL_ALPHA}-REP_ALPHA${RALPHA}-CE_ALPHA${CE_ALPHA}-STRGY${STRGY}

CUDA_VISIBLE_DEVICES=$GPU python dynamic_objective.py --strategy $STRGY --logging_dir $OUTPUT_DIR  \
  --model_name_or_path $MODEL-uncased  \
  --teacher $TEACHER_MODEL_PATH  \
  --student_num_layers $SL  --warmup_ratio 0.1  \
  --kd_kl_alpha $KL_ALPHA --kd_rep_alpha $RALPHA  \
  --ce_alpha $CE_ALPHA \
  --seed $SEED \
  --task_name $TASK_NAME \
  --fp16  \
  --do_eval --do_train  \
  --max_seq_length 128 \
  --per_device_train_batch_size 32  \
  --per_device_eval_batch_size 64  --overwrite_output_dir \
  --save_total_limit 1 \
  --logging_steps 500   \
  --evaluation_strategy steps \
  --learning_rate $LR \
  --num_train_epochs $EPOCH  \
  --output_dir $OUTPUT_DIR
done
done
done
done