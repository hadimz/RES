import random
import os
import shutil

def sample_selection_with_explanations_gender(n_smaple_with_label, path_to_attn, label_ratio = 1, args=None):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_men = {}
    path_to_attn_women = {}
    source_dir_path = './gender_data/train'
    # before selection, let's create two pools for men and women separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/men/' + path):
            path_to_attn_men[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/women/' + path):
            path_to_attn_women[path] = path_to_attn[path]
        # else:
        #     print('Something wrong with this image:', path)

    print('Total number of explanation labels in train set - men:', len(path_to_attn_men))
    print('Total number of explanation labels in train set - women:', len(path_to_attn_women))
    random.seed(args.random_seed)
    sample_paths_men = random.sample(list(path_to_attn_men), n_smaple_with_label)
    sample_paths_women = random.sample(list(path_to_attn_women), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_men:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_women:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './gender_data/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/men')
        os.mkdir(fw_dir_path + '/women')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/men/' + path):
                src = source_dir_path + '/men/' + path
                dst = fw_dir_path + '/men/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/women/' + path):
                src = source_dir_path + '/women/' + path
                dst = fw_dir_path + '/women/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)

def sample_selection_with_explanations_places(n_smaple_with_label, path_to_attn, label_ratio = 1, args=None):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_nature = {}
    path_to_attn_urban = {}
    source_dir_path = './places/train'
    # before selection, let's create two pools for nature and urban separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/nature/' + path):
            path_to_attn_nature[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/urban/' + path):
            path_to_attn_urban[path] = path_to_attn[path]
        else:
            print('Something wrong with this image:', path)
    print('Total number of explanation labels in train set - nature:', len(path_to_attn_nature))
    print('Total number of explanation labels in train set - urban:', len(path_to_attn_urban))

    random.seed(args.random_seed)
    sample_paths_nature = random.sample(list(path_to_attn_nature), n_smaple_with_label)
    sample_paths_urban = random.sample(list(path_to_attn_urban), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_nature:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_urban:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './places/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/nature')
        os.mkdir(fw_dir_path + '/urban')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/nature/' + path):
                src = source_dir_path + '/nature/' + path
                dst = fw_dir_path + '/nature/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/urban/' + path):
                src = source_dir_path + '/urban/' + path
                dst = fw_dir_path + '/urban/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)

def sample_selection_with_explanations_sixray(n_smaple_with_label, path_to_attn, label_ratio = 1, args=None):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_neg = {}
    path_to_attn_pos = {}
    source_dir_path = './sixray/train'
    # before selection, let's create two pools for men and women separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/neg/' + path):
            path_to_attn_neg[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/pos/' + path):
            path_to_attn_pos[path] = path_to_attn[path]
        else:
            print('Something wrong with this image:', path)

    print('Total number of explanation labels in train set - negative:', len(path_to_attn_neg))
    print('Total number of explanation labels in train set - positive:', len(path_to_attn_pos))
    random.seed(args.random_seed)
    sample_paths_pos = random.sample(list(path_to_attn_pos), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_pos:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './sixray/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/neg')
        os.mkdir(fw_dir_path + '/pos')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/neg/' + path):
                src = source_dir_path + '/neg/' + path
                dst = fw_dir_path + '/neg/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/pos/' + path):
                src = source_dir_path + '/pos/' + path
                dst = fw_dir_path + '/pos/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)

