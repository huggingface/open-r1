task=Llama-3.2-3B-Instruct_backtrack_suffix_demo
data=hf-cmu-collab/metaMATH-Llama-3.1-8B-Instruct-GRPO
model=hf-cmu-collab/Llama-3.2-3B-Instruct_backtrack_suffix_iteration1
epoch=1
folder=backtrack-rl

alphas=(
    0 
    0.2
    0.4
    0.8
    1.6
)

final_reward_weights=(
    "zero"
    "uniform"
    "tts"
    "ttf"
)

for alpha in "${alphas[@]}"; do
    for weight in "${final_reward_weights[@]}"; do
        task=Llama-3.2-3B-Instruct_backtrack_grpo_${alpha}_${weight}
        accelerate launch --config_file grpo-metaMATH/configs/zero3.yaml grpo-metaMATH/src/open_r1/grpo.py \
            --model_name_or_path ${model} \
            --dataset_name ${data} \
            --dataset_start 0 \
            --dataset_end 20000 \
            --reward_funcs "final" "info_gain" \
            --alpha ${alpha} \
            --final_reward_weight ${weight} \
            --learning_rate 1.0e-6 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --weight_decay 0.01 \
            --num_train_epochs ${epoch} \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 32 \
            --max_prompt_length 1500 \
            --max_completion_length 1024 \
            --num_generations 8 \
            --use_vllm True \
            --vllm_gpu_memory_utilization 0.8 \
            --temperature 0.9 \
            --logging_steps 1 \
            --eval_strategy no \
            --save_strategy "steps" \
            --save_steps 20 \
            --save_total_limit 8 \
            --output_dir /raid0/yqu/models/${folder}/${task}_tmp \
            --report_to wandb \
            --bf16 \
            &> /home/ubuntu/yqu/workspace/projects/finetune/output/backtrack-rl/${task}.log
    done
done

# --eval_strategy steps \
# --eval_steps 1 \
