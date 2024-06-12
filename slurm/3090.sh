#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8

# N30区NCCL跨节点通信
export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export CL_METHOD=PP

echo $SLURM_JOB_NODELIST
nvidia-smi

srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=testpp --kill-on-bad-exit=1 \
	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/deepspeed \
		--master_port $MASTER_PORT \
		/HOME/scz0cqe/run/LLMCL/main.py  \
			--data_path /HOME/scz0cqe/run/LLMCL/data_files \
			--dataset_name "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
			--model_name_or_path /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
			--per_device_train_batch_size 2 \
			--per_device_eval_batch_size 2 \
			--gradient_accumulation_steps 1 \
			--max_prompt_len 512 \
			--max_ans_len 512 \
			--learning_rate 1e-4 \
			--weight_decay 0. \
			--num_train_epochs 5 \
			--warmup_ratio 0.2 \
			--seed 42 \
			--adapter prompt \
			--cl_method $CL_METHOD \
			--resume_from_checkpoint ./outputs/models/$CL_METHOD \
			--output_dir ./outputs/models/$CL_METHOD >./outputs/models/$CL_METHOD/log_${SLURM_JOB_ID}.txt 2>&1

# --chdir /HOME/scz0cqe/run/LLMCL