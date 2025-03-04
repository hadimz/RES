#!/bin/bash
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 0 --model_name mine_gender_data_weak_feedback_seed_0_progressive_n_50
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 0 --model_name mine_gender_data_weak_feedback_seed_1_progressive_n_50
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 0 --model_name mine_gender_data_weak_feedback_seed_2_progressive_n_50
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 3 --mine true --eta 0 --model_name mine_gender_data_weak_feedback_seed_3_progressive_n_50
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 4 --mine true --eta 0 --model_name mine_gender_data_weak_feedback_seed_4_progressive_n_50


