from data import *
import os, os.path
import numpy as np
from numpy import random
import fnmatch
import shutil

"""
The given 2 datasets are used to train 2 models repectively

"""
def scan_file(file_dir = '', file_postfix = 'png'):
    '''
    This function will scan the file in the given directory and return the number
    and file name list for files satisfying the postfix condition.
    :param file_dir: string, should end with '/';
    :param file_type: string, no need for '.';
    :return: file_count: list of file names whose postfix satisfies the condition;
    '''
    file_count = 0
    file_list = []
    for f_name in os.listdir(file_dir):
        if fnmatch.fnmatch(f_name, ('*.' + file_postfix)):
            file_count += 1
            file_list.append(f_name)
    return file_count, file_list

def split_img_mask(general_dir='', item = '', img_dir='', mask_dir='', txt_dir=''):
    """
    split the dataset into img and mask
    and save the img and mask into different files
    """
    txt = item.replace('/', '.txt')
    print('-------------------start scanning-------------------')
    original_dir = general_dir+item
    num, names = scan_file(original_dir)
    count1 = 0
    count2 = 0
    for name in names:
        if 'mask' in name:
            # new_name = (item + name).replace('/','') # to faciliate the concatenation with the ref imgs
            # shutil.copy(os.path.join(original_dir,name), os.path.join(mask_dir, new_name))
            shutil.copy(os.path.join(original_dir,name), os.path.join(mask_dir, name))
            count1 += 1
        else:
            write_record(txt_dir, txt, img_dir+name)
            # new_name = (item + name).replace('/','') # to faciliate the concatenation with the ref imgs
            # shutil.copy(os.path.join(original_dir,name), os.path.join(img_dir, new_name))
            shutil.copy(os.path.join(original_dir,name), os.path.join(img_dir, name))
            count2 += 1
    print("no of masks: ", count1)
    print("no of imgs:  ", count2)
    if count1 + count2 == num:
        print(general_dir, item)
        print('--------------all transferred to train--------------\n')
    
def write_record(txt_dir, txt, name):
    path = os.path.join(txt_dir, txt)
    f = open(path, "a+")
    f.write(name+'\n')
    f.close()


if __name__ == "__main__":

    general_dir = "../RR/AI_CV_DATA/缺陷检测/"
    sub_dir1 = "mint_口香糖盒子外部/train_DF124-Opt2-20190604-0627/bg/"
    sub_dir2 = "mint_口香糖盒子外部/train_DF124-Opt2-20190604-0627/ng/"
    # sub_dir3 is used when doing the 6-channel train and test
    sub_dir3 = "mint_口香糖盒子外部/train_DF124-Opt2-20190604-0627/ref/"

    """the following will be used when PIE data is used"""
    # sub_dir3 = "PIE盒子/DF1_OPT3_20190715_0724/bg"
    # sub_dir4 = "PIE盒子/DF1_OPT3_20190715_0724/ng"
    # sub_dir5 = "PIE盒子/DF2_3_OPT3_20190715_0725/bg"
    # sub_dir6 = "PIE盒子/DF2_3_OPT3_20190715_0725/ng"

    mint1 = "mint1/"
    mint2 = "mint2/"
    mint3 = "mint3/"
    mints = [mint1, mint2, mint3]

    sub_dir = [sub_dir1, sub_dir2]
    dirs = [general_dir + sub for sub in sub_dir]

    # generate txt files to record which imgs are mint1,2,3...
    txt_dir = "data/RR_mint/"
    txt1 = "mint1.txt"
    txt2 = "mint2.txt"
    txt3 = "mint3.txt"

    save_img_dir = "data/RR_mint/train/image/"
    save_mask_dir = "data/RR_mint/train/mask/"
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)

    # num1, file_list1 = scan_file(dirs[0])
    # print('bg mint1 ', num1)
    # num2, file_list2 = scan_file(dirs[1])
    # print('bg mint2 ', num2)
    # num3, file_list3 = scan_file(dirs[2])
    # print('bg mint3 ', num3)
    # num4, file_list4 = scan_file(dirs[3])
    # print('ng mint1 ', num4)
    # num5, file_list5 = scan_file(dirs[4])
    # print('ng mint2 ', num5)
    # num6, file_list6 = scan_file(dirs[5])
    # print('ng mint3 ', num6)

    # split mask and img, first saving all to the training folder
    for item in dirs:
        for mint in mints:
            split_img_mask(item, mint, save_img_dir, save_mask_dir, txt_dir)

    # split train and test
    total_num, total_list = scan_file(save_img_dir)
    # set the fraction of test files among all the files
    test_frac = 0.2
    test_img_dir = "data/RR_mint/test/image"
    test_mask_dir = "data/RR_mint/test/mask"
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)
    if not os.path.exists(test_mask_dir):
        os.makedirs(test_mask_dir)
    test_num = int(total_num*test_frac)
    print("------------randomly generating testing files-------------")
    print("test num: ", test_num)
    print("train num: ", total_num - test_num)
    random.seed(1)
    for i in range(test_num):
        img_name = random.choice(os.listdir(save_img_dir)) # choose random img from the folder
        mask_name = img_name.replace('.png', '_mask.png') # find the corresponding mask
        shutil.move(os.path.join(save_img_dir, img_name), os.path.join(test_img_dir, img_name))
        shutil.move(os.path.join(save_mask_dir, mask_name), os.path.join(test_mask_dir, mask_name))
    
    print("-------------------test files genereated-------------------")

