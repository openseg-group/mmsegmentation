import os
import numpy as np

from os import listdir
from os.path import isfile, join

def dir2file(path="/home", file="/home/train.txt"):
    assert path != None and file != None
    with open(file, 'w') as f:
        for _ in listdir(path):
            if isfile(join(path, _)):
                print(_)
                f.write(_.replace('_leftImg8bit.png', '') + '\n')

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
    # dir2file(path='/home/yuhui/teamdrive/dataset/cityscapes/train/image', file='/home/yuhui/teamdrive/dataset/cityscapes/train.txt')
    # dir2file(path='/home/yuhui/teamdrive/dataset/cityscapes/coarse/image', file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt')

    ''' sample a set of unlabeled images '''
    random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v1.txt', sample_num=3000)
    random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v2.txt', sample_num=3000)
    random_sample(file='/home/yuhui/teamdrive/dataset/cityscapes/coarse.txt', sub_file='/home/yuhui/teamdrive/dataset/cityscapes/coarse3k_v3.txt', sample_num=3000)



