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
export CL_METHOD=ER
export OUTPUT_DIR=./outputs/models/${CL_METHOD}_new
if [ ! -d $OUTPUT_DIR ]
then 
	mkdir -p $OUTPUT_DIR
fi
echo $SLURM_JOB_NODELIST
nvidia-smi
export SLURM_OVERLAP=1
# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=testpp --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/deepspeed \
# 		--master_port $MASTER_PORT \
# 		/HOME/scz0cqe/run/LLMCL/main.py  \
# 			--data_path /HOME/scz0cqe/run/LLMCL/data_files \
# 			--dataset_name "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
# 			--model_name_or_path /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--per_device_train_batch_size 2 \
# 			--per_device_eval_batch_size 2 \
# 			--gradient_accumulation_steps 1 \
# 			--max_prompt_len 512 \
# 			--max_ans_len 512 \
# 			--learning_rate 1e-4 \
# 			--weight_decay 0. \
# 			--num_train_epochs 5 \
# 			--warmup_ratio 0.2 \
# 			--seed 42 \
# 			--adapter lora \
# 			--cl_method $CL_METHOD \
# 			--resume_from_checkpoint $OUTPUT_DIR \
# 			--output_dir $OUTPUT_DIR >$OUTPUT_DIR/log_${SLURM_JOB_ID}.txt 2>&1

# --chdir /HOME/scz0cqe/run/LLMCL
export CUDA_VISIBLE_DEVICES=0,1,2,3
srun  --gres=gpu:4 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa" \
			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_medmcqa >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_medmcqa/infer_log_${SLURM_JOB_ID}.txt 2>&1 &


sleep 20
export CUDA_VISIBLE_DEVICES=4,5,6,7
srun  --gres=gpu:4 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten" \
			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_20Minuten >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_20Minuten/infer_log_${SLURM_JOB_ID}.txt 2>&1 &

# sleep 180

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_NumGLUE-cm >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_NumGLUE-cm/infer_log_${SLURM_JOB_ID}.txt 2>&1

# sleep 20

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_ScienceQA >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_ScienceQA/infer_log_${SLURM_JOB_ID}.txt 2>&1



# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_MeetingBank >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_MeetingBank/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
# 			--dataset_names "C-STANCE,FOMC" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_FOMC >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_FOMC/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER \
# 			--dataset_names "C-STANCE" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_C-STANCE >/HOME/scz0cqe/run/LLMCL/outputs/models/ER/ER_lora_checkpoint_C-STANCE/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_C-STANCE >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_C-STANCE/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_FOMC >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_FOMC/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_MeetingBank >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_MeetingBank/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_ScienceQA >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_ScienceQA/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_NumGLUE-cm >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_NumGLUE-cm/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_20Minuten >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_20Minuten/infer_log_${SLURM_JOB_ID}.txt 2>&1

# srun --gres=gpu:8 --chdir /HOME/scz0cqe/run/LLMCL --job-name=infER --kill-on-bad-exit=1 \
# 	/HOME/scz0cqe/run/.conda/envs/LLMCL/bin/python \
# 		/HOME/scz0cqe/run/LLMCL/acc_inference.py \
# 			--model_name /HOME/scz0cqe/run/PLLM/llama-2-7b-chat-hf \
# 			--output_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new \
# 			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa" \
# 			--adapter_checkpoint_dir /HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_medmcqa >/HOME/scz0cqe/run/LLMCL/outputs/models/ER_new/ER_lora_checkpoint_medmcqa/infer_log_${SLURM_JOB_ID}.txt 2>&1

wait
echo 'all down'
