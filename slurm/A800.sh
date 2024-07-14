#!/bin/bash

export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export CL_METHOD=pp

# ilora

export OUTPUT=./outputs/models/$CL_METHOD
if [ ! -d $OUTPUT ] 
then
	mkdir -p $OUTPUT
fi

/home/rtxrtx/envs/LLMCL/bin/deepspeed \
		--master_port $MASTER_PORT \
        --include=localhost:0,1,2,3,4,5 \
		/home/rtxrtx/github/LLMCL/main.py  \
			--data_path /home/rtxrtx/github/LLMCL/data_files \
			--dataset_name "C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,20Minuten,medmcqa,jecqa" \
			--model_name_or_path /home/rtxrtx/PLLM/llama-2-7b-hf \
			--per_device_train_batch_size 2 \
			--per_device_eval_batch_size 2 \
			--gradient_accumulation_steps 2 \
			--max_prompt_len 512 \
			--max_ans_len 512 \
			--learning_rate 1e-4 \
			--weight_decay 0. \
			--num_train_epochs 5 \
			--warmup_ratio 0.2 \
			--seed 42 \
			--adapter prompt \
			--cl_method $CL_METHOD \
			--output_dir ./outputs/models/$CL_METHOD >./outputs/models/$CL_METHOD/log_$(date +%Y_%m_%d_%H_%M).txt 2>&1 &

# 			--resume_from_checkpoint ./outputs/models/$CL_METHOD \