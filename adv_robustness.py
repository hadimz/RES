import argparse
import os
from os import walk
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from autoattack import AutoAttack

import warnings
warnings.filterwarnings("ignore")

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--data_dir', default='gender_data', type=str)
    parser.add_argument('--model_dir', type=str, default='./model_save/',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--model_name', type=str, default='model_out',
                        help='The model filename that will be used for evaluation or phase 2 fine-tuning.')
    parser.add_argument('--test-batch', default=1000, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('--epsilon', default=0.03, type=float,
                        help='Slack factor for robust attention loss')
    
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args
args = get_args()

if __name__ == '__main__':
    # Data loading code
    testdir = os.path.join(args.data_dir, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch,
        shuffle=False,
        pin_memory=True)

    
    # perform adversarial robustness evaluation using AutoAttack
    
    epsilon = args.epsilon
    log_file = "logfile.txt"
    print(f'model dir: {args.model_dir}')
    for (dirpath, dirnames, model_fns) in walk(args.model_dir):
            for model_fn in sorted(model_fns, key=str.casefold):
                # load the model we want to evaluate
                model = torch.load(os.path.join(dirpath, model_fn))
                if args.use_cuda:
                    model.cuda()
                adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', log_path=log_file)
                adversary.attacks_to_run = ['apgd-ce', 'square']
                with open(log_file, 'a') as file:
                    file.write(f'{model_fn:50}')
                for batch_idx, (images, labels, _) in enumerate(test_loader):
                    print(f'adversarial robustness evaluation with epsilon {epsilon}, model: {model_fn} ')
                    adversary.run_standard_evaluation_individual(images, labels, bs=128)
