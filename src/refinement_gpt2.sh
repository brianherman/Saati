#!/bin/bash
python run_language_modeling.py --output_dir=output --model_type=gpt2 --model_name_or_path=D:\Development\models\gpt2-medium --do_train --train_data_file=./twitter_train.txt --do_eval --eval_data_file=./twitter_test.txt --per_device_train_batch_size=1 --overwrite_output_dir
