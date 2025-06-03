#!/usr/bin/env sh

train_data_path='./configs/data.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=checkpoints
batch_size=4
snr_type=lognorm
lrn=8e-5
lr=8e-5
precision=bf16
size=1024

exp_name=${model}_bs${batch_size}_lr${lrn}_${precision}_NewBee
mkdir -p results/"$exp_name"

NNODES=1
NPROC_PER_NODE=4
MASTER_PORT=12345 #1234
NODE_RANK=0

torchrun --nproc_per_node=4 \
         --master_port=12345 \
         --master_addr=localhost \
    finetune-g3.py \
    --master_port 18182 \
    --global_bsz_${size} 20 \
    --micro_bsz_${size} 5 \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} --grad_precision bf16 --qk_norm \
    --global_seed 20241207 \
    --num_workers 4 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    2>&1 | tee -a results/"$exp_name"/output.log
