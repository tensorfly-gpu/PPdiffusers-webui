import os

os.system('python modules/train_dreambooth_lora.py\
  --pretrained_model_name_or_path= "Baitian/momocha"  \
  --instance_data_dir="./Xinhai" \
  --output_dir="./dream_booth_lora_outputs" \
  --instance_prompt="Xinhai" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --lora_rank=128 \
  --validation_prompt="Xinhai" \
  --validation_epochs=25 \
  --validation_guidance_scale=5.0 \
  --use_lion False \
  --seed=0')