#!/bin/bash

export NCCL_P2P_DISABLE=1
ulimit -n 65536



PROJECT_CACHE=your folder
work_dir=project folder
model=mistral3b

n_epochs=2
batch_size=64
sparsity_ratio=0.0
is_eval=is_eval
#is_eval=none


cd $work_dir




bash ./scripts/codealpaca_r_32.sh  ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5

bash ./scripts/codealpaca_r_64.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5



bash ./scripts/gsm8k_r_32.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5

bash ./scripts/gsm8k_r_64.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5


bash ./scripts/saferpaca_r_32.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5

bash ./scripts/saferpaca_r_64.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5


bash ./scripts/nlu_r_32.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5

bash ./scripts/nlu_r_64.sh ${PROJECT_CACHE} ${model} ${n_epochs} ${batch_size}  ${sparsity_ratio}  ${is_eval}
sleep  5