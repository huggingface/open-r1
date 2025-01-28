task=Llama-3.2-3B-Instruct_backtrack_suffix_grpo
data=hf-cmu-collab/metaMATH_0_20000_v05.00_suffix_processed_new
model=/raid0/yqu/models/backtrack/Llama-3.2-3B-Instruct_metaMATH_t0_n1_backtrack_with_shared_prefix_implicit_backtrack_0113
model_revision=5923657491414a868c6abed3bf789653eeeed164
epoch=1
folder=backtrack-rl
source /home/ubuntu/miniconda3/bin/activate
conda activate openr1

export HF_HOME=/raid0/yqu/huggingface
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/trl/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/yqu/workspace/projects/hf-cmu-collab"
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/yqu/workspace/projects/hf-cmu-collab/evaluation/scripts"

export WANDB_API_KEY="315e959a539374dbcc7d86cec33ede7187be943f"
export WANDB_NAME=${task}
export WANDB_ENTITY=yuxiao98
export WANDB_PROJECT=${folder}


accelerate launch --config_file /home/ubuntu/yqu/workspace/projects/open-r1/configs/zero3.yaml /home/ubuntu/yqu/workspace/projects/open-r1/src/open_r1/grpo.py \
    --model_name_or_path ${model} \
    --dataset_name ${data} \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_prompt_length 1500 \
    --max_completion_length 1024 \
    --num_generations 4 \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 8 \
    --output_dir /raid0/yqu/models/${folder}/${task} \
    --report_to wandb \
    --bf16 \
    &> /home/ubuntu/yqu/workspace/projects/finetune/output/backtrack-rl/${task}.log

# --eval_strategy steps \
# --eval_steps 1 \
