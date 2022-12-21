import mmcv
import numpy as np
import shutil
import os
import xml.etree.ElementTree as ET


def get_X_L_0(cfg, X_L_0_size):
    # load all image name
    name = load_name_list(cfg.data.train.X_L_file)
    # get all indexes
    X_all = np.arange(len(name))
    # randomly select labeled set
    np.random.shuffle(X_all)
    X_L = X_all[:X_L_0_size].copy()
    X_L.sort()
    X_U = np.array(list(set(X_all) - set(X_L)))
    X_U.sort()
    X_all.sort()
    # return X_L, X_U, X_all, anns
    return X_L, X_U, X_all, name


def get_X_L_cycle(path, all_image_name):
    # load dataset anns
    name = load_name_list(path)
    X_L = np.arange(len(name))
    for i, data in enumerate(name):
        idx = np.where(data == all_image_name)[0][0]
        X_L[i] = idx
    return X_L


def create_X_L_file(cfg, X_L, X_U, all_image_name):
    # create labeled ann files
    X_L_path = cfg.work_dir + '/trainval_X_L' + '.txt'
    np.savetxt(X_L_path, all_image_name[X_L], fmt='%s')
    X_U_path = cfg.work_dir + '/trainval_X_U' + '.txt'
    np.savetxt(X_U_path, all_image_name[X_U], fmt='%s')
    cfg.data.train.X_L_file = X_L_path
    return cfg


def copy_ann_file(cfg, X_L, all_image_name):
    image_name = all_image_name[X_L]
    data_root = cfg.data.test.ann_file
    des_root = cfg.work_dir[:-6] + 'annfile/'
    if not os.path.exists(des_root):
        os.makedirs(des_root)
    if isinstance(image_name, str):
        shutil.copyfile(data_root + image_name + '.txt', des_root + image_name + ".txt")
    else:
        for name in image_name:
            shutil.copyfile(data_root + name + '.txt', des_root + name + ".txt")


def copy_ann_file_partial(cfg, i, all_image_name, save_num):
    # read
    name = all_image_name[i]

    data_root_file = cfg.data.test.ann_file + name + '.txt'
    instance_info_list = open(data_root_file).readlines()   # list(str)

    des_root_file = cfg.work_dir[:-6] + 'annfile/' + name + ".txt"
    if not os.path.exists(des_root_file):
        os.system(r"touch {}".format(des_root_file))

    # write
    num = 0
    with open(des_root_file, 'a') as f1:
        for instance_info in instance_info_list:
            f1.write(instance_info)
            num += 1
            if num == save_num:
                break


def modify_partial_data_set(cfg):
    cfg.data.train.X_L_file = cfg.work_dir + '/annfile/trainval_X_L.txt'
    ori_ann_file = cfg.data.train.ann_file
    cfg.data.train.ann_file = cfg.work_dir + '/annfile/queried/'
    return ori_ann_file


def load_name_list(paths):
    name = np.loadtxt(paths, dtype='str')
    return name




