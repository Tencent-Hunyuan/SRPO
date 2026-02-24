NNODES=$HOST_NUM

NPROC_PER_NODE=$HOST_GPU_NUM
NODE_RANK=$INDEX
DATA_NUM_WORKERS=4
MODEL_TYPE="FLUX"
SP_SIZE=1

mkdir images


torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_endpoint $CHIEF_IP:29601 \
    --rdzv_id 456 \
     fastvideo/SRPO_Qwen.py \
    --seed 42 \
    --pretrained_model_name_or_path   \
    --vae_model_path  \
    --data_json_path  \
    --output_dir  \
    --offline_dict  \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 30000000 \
    --learning_rate 3e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --offline_end '-1' \
    --allow_tf32 \
    --train_cfg 1 \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 26 \
    --image_p '5' \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 50 \
    --weight_decay 0.0001 \
    --shift 1 \
    --ignore_last \
    --train_timestep 5 40 \
    --timestep_length 100 \
    --vis_size 1328 \
    --groundtruth_ratio 0.6 \
    --reward_model PickScore \
   

