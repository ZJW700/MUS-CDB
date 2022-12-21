# MUS-CDB

![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![PyTorch 1.8](https://img.shields.io/badge/PyTorch-1.8-EE4C2C.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/CUDA-11.1-green.svg?style=plastic)
![cuDNN 7.6.5](https://img.shields.io/badge/cudnn-7.6.5-green.svg?style=plastic)
[![LICENSE](https://img.shields.io/github/license/yuantn/mi-aod.svg)](https://github.com/yuantn/mi-aod/blob/master/LICENSE)

<!-- TOC -->

- [Abstract](#abstract)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Repository Contributor](#repository-contributor)
- [License](#license)
- [Citation](#citation)

<!-- TOC -->


PyTorch implementation of [***MUS-CDB: Mixed Uncertainty Sampling with Class Distribution Balancing for Active Annotation in Aerial Object Detection***]().

## Abstract
![Task](framework.jpg)
Recent aerial object detection models rely on a large amount of labeled training data, which requires unaffordable manual labeling costs in large aerial scenes with dense objects. Active learning is effective in reducing the data labeling cost by selectively querying the informative and representative unlabelled samples. However, existing active learning methods are mainly with class-balanced setting and image-based querying for generic object detection tasks, which are less applicable to aerial object detection scenario due to the long-tailed class distribution and dense small objects in aerial scenes. In this paper, we propose a novel active learning method for cost-effective aerial object detection. Specifically, both object-level and image-level informativeness are considered in the object selection to refrain from redundant and myopic querying. Besides, an easy-to-use class-balancing criterion is incorporated to favor the minority objects to alleviate the long-tailed class distribution problem in model training. To fully utilize the queried information, we further devise a training loss to mine the latent knowledge in the undiscovered image regions. Extensive experiments are conducted on the DOTA-v1.0 and DOTA-v2.0 benchmarks to validate the effectiveness of the proposed method. The results show that it can save more than 75\% of the labeling cost to reach the same performance compared to the baselines and state-of-the-art active object detection methods. 


## Installation
### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n mus-cdb python=3.7 -y
    conda activate mus-cdb
    ```

2. Please install **PyTorch 1.8.0** and **torchvision 0.9.0** for **CUDA 11.1** following the [official instructions](https://pytorch.org/get-started/previous-versions/#v160), e.g.,

    ```shell
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
3. Please install mmcv-full, we recommend you to install the pre-build package as below.
    ```shell
    pip install mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    ```

4. Please clone the MUS-CDB repository.

    ```shell
    git clone https://github.com/zjw700/MUS-CDB.git
    cd MUS-CDB
    ```
   
5. Please download the required package.

    ```shell
    pip install -r requirements/build.txt
    ```
   
## Data Preparation

Please refer to [data_preparation.md](data_preparation.md) for dataset installation and segmentation.


<!-- You may also use the following commands directly:

```bash
cd $YOUR_DATASET_PATH
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
``` -->


## Train and Test
Please modify the corresponding pretrained model path, they are located in:

```python
Line 16 of configs/redet/redet_re50_refpn_1x_dota_le90.py: pretrained='$PRETRAINED_MODEL_PATH/'
Line 16 of configs/redet/redet_re50_refpn_1x_dota2_le90.py: pretrained='$PRETRAINED_MODEL_PATH/'
```

We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

If you use only a single GPU, you can use the `script.sh` file directly as below:

```bash
chmod 700 ./script.sh
./script.sh 
```

The script uses the train.py and test.py functions to train and test where the value of `config_path`,  `sample`, `c_0_path`, `s_0_path` and `work_dir` shound be changed according to your actual situation (e.g., `config_path` should be replaced by the path of the config file in the `configs/_base_` folder).

 Please note more parameters can be found and mofidy in the file of train.py and test.py .


## Model Zoo

The trained model for each cycle in active learning are [Baidu Drive (Extraction code:ASLK)](https://pan.baidu.com/s/1aDvn4lrIwdF5jsT1NveaoA).


## License
This project is released under the [Apache 2.0 license](./LICENSE).

## Citation

If you find this repository useful for your publications, please consider citing our [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yuan_Multiple_Instance_Active_Learning_for_Object_Detection_CVPR_2021_paper.pdf).

```bibtex
@misc{2212.02804,
Author = {Dong Liang and Jing-Wei Zhang and Ying-Peng Tang and Sheng-Jun Hang},
Title = {MUS-CDB: Mixed Uncertainty Sampling with Class Distribution Balancing for Active Annotation in Aerial Object Detection},
Year = {2022},
Eprint = {arXiv:2212.02804},
}
```