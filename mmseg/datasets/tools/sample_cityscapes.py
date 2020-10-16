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

def dir2file(path="/home", file="/home/train.txt"):
    assert path != None and file != None
    all_files = getListOfFiles(path)
    with open(file, 'w') as f:
        for _ in all_files:
            if isfile(_):# join(path, _)
                print(_)
                f.write(_.replace('_leftImg8bit.png', '').
                        replace('/home/yuhui/teamdrive/dataset/original_cityscapes/leftImg8bit/train/', '') + '\n')

def random_sample(file="/home/train.txt", sub_file="/home/subset.txt", sample_num=3000):
    with open(file, 'r') as f:
        samples = f.read().splitlines()
    np.random.shuffle(samples)

    selected_list = samples[:sample_num]
    with open(sub_file, 'w') as f:
        for _ in selected_list:
            f.write(_.replace('_leftImg8bit.png', '') + '\n')

if __name__ == '__main__':
    ''' generate the txt files for train set + coarse set '''
    dir2file(path='/home/yuhui/teamdrive/dataset/original_cityscapes/leftImg8bit/train', file='/home/yuhui/teamdrive/dataset/original_cityscapes/train.txt')
    # dir2file(path='/home/yuhui/teamdrive/dataset/cityscapes/coarse/image', file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt')

    ''' sample a set of unlabeled images '''
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v1.txt', sample_num=3000)
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v2.txt', sample_num=3000)
    # random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v3.txt', sample_num=3000)



