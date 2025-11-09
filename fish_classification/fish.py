###################################################################################################
#
# Copyright (C) 2025 (Your Name)
# Adapted from Cats and Dogs Dataset Loader by Maxim Integrated
#
###################################################################################################
"""
Fish Dataset Loader
Dataset: bangus, gold_fish, gourami
"""

# Essential Imports
import os
import torch
import torchvision
from torchvision import transforms
import ai8x

# Custom Imports
from PIL import Image
import errno
import shutil
import sys
import numpy


torch.manual_seed(0)

def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    if orig_img.mode != 'RGB':
        orig_img = orig_img.convert('RGB')
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
    ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """
    Augment with center crop and blurring
    """
    if orig_img.mode != 'RGB':
        orig_img = orig_img.convert('RGB')
        
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
    ])
    return train_transform(orig_img)


def fish_get_datasets(data, load_train=True, load_test=True, aug=2):
    """
    Load Fish Dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "fish")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        print("******************************************")
        print("Dataset not found!")
        print("Please ensure the dataset is in this structure:")
        print("  data/fish/train/bangus/")
        print("  data/fish/train/gold_fish/")
        print("  data/fish/train/gourami/")
        print("  data/fish/test/bangus/")
        print("  data/fish/test/gold_fish/")
        print("  data/fish/test/gourami/")
        print("******************************************")
        sys.exit("Fish dataset not found!")
    else:
        processed_dataset_path = os.path.join(dataset_path, "augmented")

        if os.path.isdir(processed_dataset_path):
            print("Augmented folder exists. Remove if you want to regenerate.")
        else:
            os.makedirs(processed_dataset_path, exist_ok=True)

            train_path = os.path.join(dataset_path, "train")
            test_path = os.path.join(dataset_path, "test")
            processed_train_path = os.path.join(processed_dataset_path, "train")
            processed_test_path = os.path.join(processed_dataset_path, "test")

            os.makedirs(processed_train_path, exist_ok=True)
            os.makedirs(processed_test_path, exist_ok=True)

            # create label folders
            for d in os.listdir(train_path):
                mk = os.path.join(processed_train_path, d)
                os.makedirs(mk, exist_ok=True)
            for d in os.listdir(test_path):
                mk = os.path.join(processed_test_path, d)
                os.makedirs(mk, exist_ok=True)

            # copy test files
            test_cnt = 0
            for (dirpath, _, filenames) in os.walk(test_path):
                print(f'Copying {dirpath} -> {processed_test_path}')
                for filename in filenames:
                    if filename.endswith(('.jpg', '.png')):
                        relsourcepath = os.path.relpath(dirpath, test_path)
                        destpath = os.path.join(processed_test_path, relsourcepath)
                        destfile = os.path.join(destpath, filename)
                        shutil.copyfile(os.path.join(dirpath, filename), destfile)
                        test_cnt += 1

            # copy and augment train files
            train_cnt = 0
            for (dirpath, _, filenames) in os.walk(train_path):
                print(f'Copying and augmenting {dirpath} -> {processed_train_path}')
                for filename in filenames:
                    if filename.endswith(('.jpg', '.png')):
                        relsourcepath = os.path.relpath(dirpath, train_path)
                        destpath = os.path.join(processed_train_path, relsourcepath)
                        srcfile = os.path.join(dirpath, filename)
                        destfile = os.path.join(destpath, filename)

                        # copy original
                        shutil.copyfile(srcfile, destfile)
                        train_cnt += 1

                        orig_img = Image.open(srcfile)

                        # crop + blur
                        aug_img = augment_blur(orig_img)
                        augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

                        # affine + jitter + blur
                        for i in range(aug):
                            aug_img = augment_affine_jitter_blur(orig_img)
                            augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                            aug_img.save(augfile)
                            train_cnt += 1

            print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')

    # Loading and normalizing train dataset
    processed_train_path = os.path.join(processed_dataset_path, "train")
    processed_test_path = os.path.join(processed_dataset_path, "test")

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=processed_train_path,
                                                         transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = torchvision.datasets.ImageFolder(root=processed_test_path,
                                                        transform=test_transform)
        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'fish',
        'input': (3, 128, 128),
        'output': ('bangus', 'gold_fish', 'gourami'),
        'loader': fish_get_datasets,
    },
]
