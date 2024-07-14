#!/bin/bash

source /home/rtxrtx/anaconda3/bin/activate LLMCL
# export CUDA_VISIBLE_DEVICES=3,4,5,6
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TASK=jecqa
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export METHOD=pp

export OUTPUT=./outputs/inference/${METHOD}
if [ ! -d $OUTPUT ] 
then
	mkdir -p $OUTPUT
fi

# /home/rtxrtx/envs/LLMCL/bin/deepspeed \
# 		--master_port $MASTER_PORT \
#         --include=localhost:1,2,3,4 \
accelerate launch --main_process_port $MASTER_PORT --num_processes 4 \
		/home/rtxrtx/github/LLMCL/acc_inference.py  \
			--dataset_names "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
			--model_name /home/rtxrtx/PLLM/llama-2-7b-hf \
			--adapter_checkpoint_dir /home/rtxrtx/github/LLMCL/outputs/models/${METHOD}/${METHOD}_prompt_checkpoint_$TASK \
			--output_dir ./outputs/inference/${METHOD} >./outputs/inference/${METHOD}/log_${TASK}_$(date +%Y_%m_%d_%H_%M).txt 2>&1 &

# "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa"

# NumGLUE-cm jecqa