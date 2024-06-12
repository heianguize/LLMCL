#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:8

# N30区NCCL跨节点通信
export NCCL_DEBUG=INFO 
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

export HOSTFILE="/home/bingxing2/home/scx6392/TRACE/TRACE/hostfile.${SLURM_JOB_ID}"

for i in `scontrol show hostnames`
do
	let k=k+1
	host[$k]=$i
	echo $k,${host[$k]}
	echo "${host[$k]} slots=8" >> $HOSTFILE
done

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CL_METHOD=PP

echo $SLURM_JOB_NODELIST
nvidia-smi

srun -N 2 --gres=gpu:8 --job-name=testpp --kill-on-bad-exit=1 \
	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/deepspeed \
		--num_nodes 2 \
		--num_gpus 8 \
		-H $HOSTFILE \
		--master_addr $MASTER_ADDR \
		--master_port=$MASTER_PORT \
		/HOME/scz0cqe/run/LLMCL/main.py  \
			--data_path /HOME/scz0cqe/run/LLMCL/data_files \
			--dataset_name "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
			--model_name_or_path /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
			--per_device_train_batch_size 1 \
			--per_device_eval_batch_size 1 \
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
			--output_dir ./outputs/models/$CL_METHOD >./outputs/models/$CL_METHOD/log_${SLURM_JOB_ID}.txt 2>&1

# --chdir /HOME/scz0cqe/run/LLMCL