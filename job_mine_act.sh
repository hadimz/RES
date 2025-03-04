#!/bin/bash
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine_act true --eta 1.0 --model_name mine_act_gender_data_lambda_1.0_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine_act true --eta 1.0 --model_name mine_act_gender_data_lambda_1.0_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine_act true --eta 1.0 --model_name mine_act_gender_data_lambda_1.0_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 3 --mine_act true --eta 1.0 --model_name mine_act_gender_data_lambda_1.0_seed_3
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 4 --mine_act true --eta 1.0 --model_name mine_act_gender_data_lambda_1.0_seed_4
