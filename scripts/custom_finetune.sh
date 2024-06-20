# should be run under TinyLLaVA_Factory directory
DATA_PATH="$PWD/../subtitles_dataset.json"
IMAGE_PATH="$PWD/.."
MODEL_MAX_LENGTH=3072
OUTPUT_DIR="$PWD/../finetuned_tinyllava"
# EPOCHS=4
# BATCH_SIZE=4
# GRADIENT_ACCUMULATION_STEPS=32
# LEARNING_RATE=2e-5
EPOCHS=$1
BATCH_SIZE=$2
GRADIENT_ACCUMULATION_STEPS=$3
LEARNING_RATE=$4

mkdir -p $OUTPUT_DIR

deepspeed --include localhost:0 --master_port 29501 tinyllava/train/custom_finetune.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version phi \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora