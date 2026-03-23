#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=$HOME/.cache/huggingface
export PROJECT_CACHE=$1
export WANDB_MODE=offline
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1


# LoRI-D training and mask extraction
dataset_name=saferpaca
model=$2
n_epochs=$3
batch_size=$4
grad_norm=1
save_every=epoch_$n_epochs
sparsity_ratio=$5
lr=5e-5
lora_rank=32
lora_alpha=64

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
    python src/eval_model.py \
        --model_name $model \
        --adapter_path $adapter_path \
        --datasets hexphi \
        --results_path $results_path \
        --sparsity_ratio $sparsity_ratio \
        --data_fraction 0.3 \
        --lora_rank $lora_rank
fi