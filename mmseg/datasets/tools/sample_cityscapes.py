import os
import numpy as np

from os import listdir
from os.path import isfile, join

import pdb

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def getListOfImages(dirName):
    allFiles = list()
    for entry in os.scandir(dirName):
        if entry.is_file() and (entry.name.endswith(".jpg") or entry.name.endswith(".png")):
            allFiles.append(entry.path)
        elif entry.is_dir():
            allFiles = allFiles + getListOfImages(entry.path)
        else:
            print(f"Neither a file, nor a dir: {entry.path}")
    return allFiles

def dir2file(path="/home", file="/home/train.txt"):
    assert path != None and file != None
    all_files = getListOfFiles(path)
    with open(file, 'w') as f:
        for _ in all_files:
            if isfile(_):# join(path, _)
                print(_)
                f.write(_.replace('_leftImg8bit.png', '').
                        replace('/home/yuhui/teamdrive/dataset/original_cityscapes/leftImg8bit/train/', '') + '\n')

def dir2file_ade20k(path="/home", file="/home/train.txt"):
    assert path != None and file != None
    all_files = getListOfImages(path)
    with open(file, 'w') as f:
        for _ in all_files:
            if isfile(_):# join(path, _)
                print(_)
                f.write(_.split('/')[-1].replace('.jpg', '') + '\n')

def dir2file_coco(path="/home", file="/home/train.txt"):
    assert path != None and file != None
    all_files = getListOfImages(path)
    pdb.set_trace()
    with open(file, 'w') as f:
        for _ in all_files:
            print(_)
            # if _[-3:] == 'jpg':
            f.write(_.split('/')[-1][:-4] + '\n')


def random_sample(file="/home/train.txt", sub_file="/home/subset.txt", sample_num=3000):
    with open(file, 'r') as f:
        samples = f.read().splitlines()
    np.random.shuffle(samples)

    selected_list = samples[:sample_num]
    with open(sub_file, 'w') as f:
        for _ in selected_list:
            f.write(_.replace('_leftImg8bit.png', '') + '\n')

def post_uniform_select(file="/home/train.txt", post_file="/home/train_post.txt"):
    with open(file, 'r') as f:
        samples = f.read().splitlines()
    with open(post_file, 'w') as f:
        for _ in samples:
            img_path, label_path = _.split()
            img_name_ext = img_path.split('/')[3]
            img_name = img_name_ext[:-4]
            f.write(img_name + '\n')

def generate_cutmix_filelist(file="/home/train.txt", sub_file="/home/subset.txt", out_file="/home/out.txt"):
    with open(file, 'r') as f:
        all_samples = f.read().splitlines()

    with open(sub_file, 'r') as f:
        selected_samples = f.read().splitlines() 
    
    remained_samples = [item for item in all_samples if item not in selected_samples]

    with open(out_file, 'w') as f:
        for _ in selected_samples:
            f.write(_ + '\n')

        for _ in remained_samples:
            f.write(_ + '\n')


def one_to_two_column(file="/home/train.txt", post_file="/home/train_post.txt"):
    with open(file, 'r') as f:
        samples = f.read().splitlines()
        
    with open(post_file, 'w') as f:
        for _ in samples:
            f.write(_ + '\t' + _ + '\n')


def cct_format(file="/home/train.txt", post_file="/home/train_post.txt"):
    with open(file, 'r') as f:
        samples = f.read().splitlines()
        
    with open(post_file, 'w') as f:
        for _ in samples:
            f.write("/JPEGImages/{}.jpg /SegmentationClassAug/{}.png\n".format(_, _))


if __name__ == '__main__':
    ''' generate the txt files for train set + coarse set '''
    # dir2file(path='/home/yuhui/teamdrive/dataset/original_cityscapes/leftImg8bit/train', file='/home/yuhui/teamdrive/dataset/original_cityscapes/train.txt')
    # dir2file_coco(path='/home/yuhui/teamdrive/dataset/cityscapes/test', file='/home/yuhui/teamdrive/dataset/cityscapes/test.txt')

    # dir2file_ade20k(path='/home/yuhui/teamdrive/dataset/ade20k/train/image', file='/home/yuhui/teamdrive/dataset/ade20k/train.txt')
    # dir2file_ade20k(path='/home/yuhui/teamdrive/dataset/coco_stuff_16k/train/image', file='/home/yuhui/teamdrive/dataset/coco_stuff_16k/train.txt')
    # dir2file_coco(path='/home/yuhui/teamdrive/dataset/coco/train2017', file='/home/yuhui/teamdrive/dataset/coco/train2017.txt')
    # dir2file_ade20k(path='/home/yuhui/teamdrive/dataset/coco_stuff_10k/train/image', file='/home/yuhui/teamdrive/dataset/coco_stuff_10k/train.txt')

    # dir2file_ade20k(path='/home/yuhui/teamdrive/dataset/coco_stuff_10k/train/image', file='/home/yuhui/teamdrive/dataset/coco_stuff_10k/train.txt')

    # one_to_two_column(file="/home/yuhui/teamdrive/dataset/mapillary-vista-v1.1/train.lst",
    #                   post_file="/home/yuhui/teamdrive/dataset/mapillary-vista-v1.1/train.txt")

    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_labeled_1-16.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/662_train_supervised.txt")
    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_unlabeled_1-16.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/662_train_unsupervised.txt")

    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_labeled_1-8.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/1323_train_supervised.txt")
    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_unlabeled_1-8.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/1323_train_unsupervised.txt")

    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_labeled_1-4.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/2646_train_supervised.txt")
    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_unlabeled_1-4.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/2646_train_unsupervised.txt")

    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_labeled_1-2.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/5291_train_supervised.txt")
    cct_format(file="/home/yuhui/teamdrive/dataset/semisup_seg_dataset/pascal_voc/subset_train_aug/train_aug_unlabeled_1-2.txt",
               post_file="/home/yuhui/teamdrive/yuyua/code/ssl/CCT/5291_train_unsupervised.txt")

    ''' sample a set of unlabeled images '''
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v1.txt', sample_num=3000)
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v2.txt', sample_num=3000)
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v3.txt', sample_num=3000)
    
    # post_uniform_select(file='/home/yuhui/teamdrive/dataset/cityscapes/train_extra_3000.txt', post_file='/home/yuhui/teamdrive/dataset/cityscapes/uniform_coarse3k.txt')

    # generate_cutmix_filelist(file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug.txt',
    #                          sub_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/subset_train_aug/train_aug_labeled_1-2.txt',
    #                          out_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug_1-2.txt')

    # generate_cutmix_filelist(file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug.txt',
    #                          sub_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/subset_train_aug/train_aug_labeled_1-4.txt',
    #                          out_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug_1-4.txt')

    # generate_cutmix_filelist(file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug.txt',
    #                          sub_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/subset_train_aug/train_aug_labeled_1-16.txt',
    #                          out_file='/home/yuhui/teamdrive/dataset/mseg_dataset/PASCAL_VOC_2012/ImageSets/SegmentationAug/train_aug_1-16.txt')

