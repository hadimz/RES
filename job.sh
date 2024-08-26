python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 0.1 --model_name mine_gender_data_lambda_0.1_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 1 --model_name mine_gender_data_lambda_1.0_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 10 --model_name mine_gender_data_lambda_10_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 100 --model_name mine_gender_data_lambda_100_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 0.1 --model_name mine_gender_data_lambda_0.1_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 1 --model_name mine_gender_data_lambda_1.0_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 10 --model_name mine_gender_data_lambda_10_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 100 --model_name mine_gender_data_lambda_100_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 0.1 --model_name mine_gender_data_lambda_0.1_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 1 --model_name mine_gender_data_lambda_1.0_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 10 --model_name mine_gender_data_lambda_10_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 100 --model_name mine_gender_data_lambda_100_seed_2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name baseline_gender_data_seed_0 --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name GRADIA_gender_data_seed_0 --trainWithMap --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name HAICS_gender_data_seed_0 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-G_gender_data_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-L_gender_data_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name baseline_gender_data_seed_1 --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name GRADIA_gender_data_seed_1 --trainWithMap --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name HAICS_gender_data_seed_1 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-G_gender_data_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-L_gender_data_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name baseline_gender_data_seed_2 --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name GRADIA_gender_data_seed_2 --trainWithMap --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name HAICS_gender_data_seed_2 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-G_gender_data_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-L_gender_data_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name baseline_places_seed_0 --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name GRADIA_places_seed_0 --trainWithMap --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name HAICS_places_seed_0 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-G_places_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-L_places_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name baseline_places_seed_1 --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name GRADIA_places_seed_1 --trainWithMap --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name HAICS_places_seed_1 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-G_places_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-L_places_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name baseline_places_seed_2 --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name GRADIA_places_seed_2 --trainWithMap --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name HAICS_places_seed_2 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-G_places_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --model_name RES-L_places_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name baseline_sixray_seed_0 --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name GRADIA_sixray_seed_0 --trainWithMap --fw_sample 50 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name HAICS_sixray_seed_0 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-G_sixray_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-L_sixray_seed_0 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 0
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name baseline_sixray_seed_1 --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name GRADIA_sixray_seed_1 --trainWithMap --fw_sample 50 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name HAICS_sixray_seed_1 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-G_sixray_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-L_sixray_seed_1 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 1
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name baseline_sixray_seed_2 --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name GRADIA_sixray_seed_2 --trainWithMap --fw_sample 50 --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name HAICS_sixray_seed_2 --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-G_sixray_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 2
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --model_name RES-L_sixray_seed_2 --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 0.1 --model_name mine_places_lambda_0.1_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 1 --model_name mine_places_lambda_1.0_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 10 --model_name mine_places_lambda_10_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 100 --model_name mine_places_lambda_100_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 0.1 --model_name mine_places_lambda_0.1_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 1 --model_name mine_places_lambda_1.0_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 10 --model_name mine_places_lambda_10_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 100 --model_name mine_places_lambda_100_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 0.1 --model_name mine_places_lambda_0.1_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 1 --model_name mine_places_lambda_1.0_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 10 --model_name mine_places_lambda_10_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir places --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 100 --model_name mine_places_lambda_100_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 0.1 --model_name mine_sixray_lambda_0.1_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 1 --model_name mine_sixray_lambda_1.0_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 10 --model_name mine_sixray_lambda_10_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 0 --mine true --eta 100 --model_name mine_sixray_lambda_100_seed_0
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 0.1 --model_name mine_sixray_lambda_0.1_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 1 --model_name mine_sixray_lambda_1.0_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 10 --model_name mine_sixray_lambda_10_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 1 --mine true --eta 100 --model_name mine_sixray_lambda_100_seed_1
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 0.1 --model_name mine_sixray_lambda_0.1_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 1 --model_name mine_sixray_lambda_1.0_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 10 --model_name mine_sixray_lambda_10_seed_2
python main.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir sixray --trainWithMap --fw_sample 50  --random_seed 2 --mine true --eta 100 --model_name mine_sixray_lambda_100_seed_2