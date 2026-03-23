#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=$HOME/.cache/huggingface
export PROJECT_CACHE=$1
export WANDB_MODE=offline
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1



# LoRI-D training and mask extraction
dataset_name=codealpaca
model=$2
n_epochs=$3
batch_size=$4
grad_norm=1
save_every=epoch_$n_epochs
sparsity_ratio=$5
lr=1e-5
lora_rank=64
lora_alpha=128

exp_name="${dataset_name}_${model}/LoRI-D_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
results_path="${PROJECT_CACHE}/${dataset_name}_${model}"

python -u src/train_lora.py \
        model=$model \
        datasets=[$dataset_name] \
        exp_name=$exp_name \
        lr=$lr \
        save_every=$save_every \
        n_epochs=$n_epochs \
        batch_size=$batch_size \
        model.fsdp_policy_mp=bfloat16 \
        fsdp_port=$MASTER_PORT \
        optimizer=AdamW \
        grad_norm_strategy=even \
        max_grad_norm=$grad_norm \
        lora_rank=$lora_rank \
        lora_alpha=$lora_alpha

if [ "$6" = "is_eval" ]; then
    accelerate launch bigcode/main.py \
            --model $model \
            --peft_model $adapter_path \
            --metric_output_path $results_path \
            --tasks humaneval \
            --temperature 0.2 \
            --n_samples 20 \
            --batch_size 10 \
            --sparsity_ratio $sparsity_ratio \
            --lora_rank $lora_rank \
            --allow_code_execution
fi
