# RES
RES implementation in Pytorch for KDD'22 paper : 'RES: A Robust Framework for Guiding Visual Explanation'

## Desciption
This codebase proivdes the necessary running environment (including the human explanation label) to train and evaluate the proposed RES model on the BBBP molecular datasets. 

## Running Environment:

Python pakage requirement:
- python==3.7.9
- pytorch==1.5.0
- torchvision==0.6.0
- opencv==4.5.0
- numpy==1.16.5
- sklearn

For the full list of pakages and corresponding version, please refer to the 'conda_list.txt'

## Data preparation

1. Download the datasets as well as our human explanation labels from google drive at:

* Gender classification dataset: [https://drive.google.com/file/d/1Mt9k5Qfcp4tYqWuq7xWhf76QHLyYPDft/view?usp=sharing](https://drive.google.com/file/d/1Mt9k5Qfcp4tYqWuq7xWhf76QHLyYPDft/view?usp=sharing)
* Scene Recognition dataset: [https://drive.google.com/file/d/1ULF6UAcg9Yvy3fa50dV8H4I9ZJFAW-SK/view?usp=sharing](https://drive.google.com/file/d/1ULF6UAcg9Yvy3fa50dV8H4I9ZJFAW-SK/view?usp=sharing)

2. Extract the files and place them in 'gender_data/' and 'places/' for Gender classification and Scene Recognition dataset, respectively.

The data in the folders are mostly self-explained by their names, but just to provide a bit more info here:

*  **Train**: this folder contains our training set
*  **Val**: this folder contains our validation set
*  **Test** : this folder contains our test set
*  **attention_label** : this folder contains all the explanation ground truth stored in csv format (which should be easily interpretable by any csv viewer). You might want to look for file name with *factual*, which stores the positive explanation maps of samples where value 1 refers to the corresponding pixel the annotators think should be included in the model explanation. The other set of files with name *counterfactual* stores the negative explanation maps where value 1 indicates the corresponding pixel the annotators are certain that it must be excluded in the explanation.

Notice that we only have partial annotation labels of the whole training set (about 20%), and in fact we did not use all the data in this folder to perform training in our work. Basically you can easily extract your own training set from the current **train** folder to only consider those samples that appear in **attention_label** folder if you want a fully annotated training set.

For more information about the dataset or experiment setup, please refer to the experimental section in the paper.

## Sample Training Scripts for Gender Classification dataset (i.e. using 100 random samples with human explanation labels)

* Baseline:
```
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name baseline --fw_sample 50 --random_seed 0
```

* GRADIA:
```
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name GRADIA --trainWithMap --fw_sample 50 --random_seed 0
```

* HAICS:
```
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name HAICS --trainWithMap --fw_sample 50 --transforms HAICS --random_seed 0
```

* RES-G:
```
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-G --trainWithMap --eta 0.1 --fw_sample 50 --transforms Gaussian --random_seed 0
```

* RES-L:
```
python RES.py --train-batch 20 --test-batch 10 --n_epoch 50 --data_dir gender_data --model_name RES-L --trainWithMap --eta 0.1 --fw_sample 50 --transforms S1 --random_seed 0
```

## Testing

e.g. test the performance of the 'baseline' model on test set:
```
python RES.py --train-batch 100 --test-batch 10 -e --model_name baseline
```

## Result Validation

Where to look for the results:
1. The overall model performance can be find directly in the program output in Console
2. For the requested data, please find the 'data.csv' in the root dir
3. Explanation visualization can be find in 'attention' folder

Below are some sample explanations visualization results on RES and other comparison methods. The model-generated explanations are represented by the heatmaps overlaid on the original image samples, where more importance is given to the area with a warmer color.

<img src="https://github.com/YuyangGao/RES/blob/main/example_figs/Fig2_re.png" alt="drawing" width="1500"/>

If any further questions, please reach out to me via email yuyang.gao@emory.edu

##

And if you find this repo useful in your research, please consider cite our paper:


    @InProceedings{gao2022res,
    title={RES: A Robust Framework for Guiding Visual Explanation},
    author={Gao, Yuyang and Sun, Tong and Bai, Guangji and Gu, Siyi and Hong, Sungsoo and Zhao, Liang},
    booktitle={Proceedings of the 28th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    month = {August},
    year = {2022},
    organization={ACM}
    }
