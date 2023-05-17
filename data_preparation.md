# Preparing DOTA Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{Xia_2018_CVPR,
author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}

@article{ding2021object,
  title={Object detection in aerial images: A large-scale benchmark and challenges},
  author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael Ying and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and others},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={44},
  number={11},
  pages={7778--7796},
  year={2021},
  publisher={IEEE}
}
```

## download dota dataset
Please download DOTA-v1.0 datasets ( trainval + test ) and DOTA-v2.0 datasets ( train + val ) from [here](https://captain-whu.github.io/DOTA/dataset.html).

The data structure is as follows:
```
├── Dotadevkit
│   ├── DOTA-v1.0
│   │   ├── trainval
│   │   │   ├──labelTxt
│   │   │   ├──images
│   │   │   ├──trainval.txt
│   │   ├── test
│   │   │   ├──labelTxt
│   │   │   ├──images
│   │   │   ├──test.txt
│   ├── DOTA-v2.0
│   │   ├── trainval
│   │   │   ├──labelTxt
│   │   │   ├──images
│   │   │   ├──trainval.txt
│   │   ├── test
│   │   │   ├──labelTxt
│   │   │   ├──images
│   │   │   ├──test.txt
```
Please note that for the DOTA-v2.0 dataset, we store the downloaded train dataset in the directory `DOTA-v2.0/trainval`, and the val dataset in the directory `DOTA-v2.0/test`.

## split dota dataset

Please crop the original images into 1024×1024 patches with an overlap of 200 by run

```shell
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval_x_0.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test_x_0.json
```

Please note that `x` represents the version number of the dataset ( i.e., 1 or 2 ).

Please change the `img_dirs`, `ann_dirs` and `save_dir` in json and `base-json` in img_split.py. 


## change root path in base config

Please modify the corresponding dataset directory, they are located in:

```python
Line 3 of configs/_base_/datasets/dotav1.py: data_root='$YOUR_DATASET_PATH/Dotadevkit/split_1024_dota1_0/'
Line 3 of configs/_base_/datasets/dotav2.py: data_root='$YOUR_DATASET_PATH/Dotadevkit/split_1024_dota2_0'

```
Please change the `$YOUR_DATASET_PATH`s above to your actual splited dataset directory.


And after that, please ensure the splited data directory tree is as below:

```
├── Dotadevkit
│   ├── split_1024_dota1_0
│   │   ├── trainval
│   │   │   ├──annfiles
│   │   │   │  ├──P0000__1024__0___0.txt
│   │   │   │      .
│   │   │   │      .
│   │   │   │      .
│   │   │   │  ├──P2805__1024__1808___2181.txt
│   │   │   ├──images
│   │   │   │  ├──P0000__1024__0___0.png
│   │   │   │      .
│   │   │   │      .
│   │   │   │      .
│   │   │   │  ├──P2805__1024__1808___2181.png
│   │   │   ├──trainval.txt
│   │   ├── test
│   │   │   ├──annfiles
│   │   │   ├──images
│   │   │   ├──test.txt
│   ├── split_1024_dota2_0
│   │   ├── trainval
│   │   ├── test
```

Please note : the splited DOTA-v2.0 datasets ( *train* ) and  ( *val* ) are separately stored in the directory '.../split_1024_dota2_0/trainval/' and '.../split_1024_dota2_0/test/'.

## create partially labeled dataset
Please create two 'annfile' files for different datasets DOTA-V1.0 and DOTA-v2.0, according to the following data structure. The 'queried' directory is initially empty, and the 'unqueried' directory stores all ground truth txt files for the training set.
```
├── annfile
│   ├── queried
│   ├── unqueried
│   │   ├──P0000__1024__0___0.txt
│   │   │      .
│   │   │      .
│   │   │      .
│   │   ├──P2805__1024__1808___2181.txt

```
When running our method 'mus-cdb', please first paste an 'annfile' in the output directory. The specific file location is shown as follows: "/data/MUS-CDB/work_dirs/mus_cdb/EXP0/annfile".
