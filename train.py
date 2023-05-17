# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import build_dataloader

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes
from tools.AL_utils.compute_image_uncertainty import *
from tools.AL_utils.active_datasets_dota import *
from mmcv.runner import load_checkpoint
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default='/PATH_TO/MUS-CDB/configs/redet/redet_re50_refpn_1x_dota_le90.py',
                        help='train config file path')
    parser.add_argument('--work-dir',
                        default='/PATH_TO/MUS-CDB/work_dirs',
                        help='the dir to save logs and models')
    parser.add_argument('--exp-num',
                        default='EXP0')
    parser.add_argument('--cycle',
                        default=0)
    parser.add_argument('--class-num',
                        default=15,
                        help="class num: DOTA-v1.0(15) and DOTA-v2.0(18)")
    parser.add_argument('--al-sample',
                        default='DGCB',
                        choices=['IGUS', 'DGCB', 'mus-cdb'],
                        )
    parser.add_argument('--X_L_0_size',
                        default=1052
                        )
    parser.add_argument('--budget',
                        default=5000
                        )
    parser.add_argument('--score_thresh',
                        default=0.10)
    parser.add_argument('--iou_thresh',
                        default=0)
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        default=0,
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        default='True',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set multi-process settings
    setup_multi_processes(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpu_ids is not None:
        cfg.gpu_ids = [args.gpu_ids]
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    logger.info(f'EXP_NUM::{args.exp_num}')
    logger.info(f'CYCLE::{args.cycle}')
    print("**" * 30)
    print("EXP_NUM::" + args.exp_num)
    print("CYCLE::" + str(args.cycle))
    print("**" * 30)

    cycle = int(args.cycle)
    cfg.work_dir = cfg.work_dir + '/' + args.exp_num + '/cycle' + str(cycle)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    label_num = torch.zeros(args.class_num)  # added

    # ==================== cycle==0 =====================================
    X_L_0_size = args.X_L_0_size
    X_L, X_U, X_all, all_image_name = get_X_L_0(cfg, X_L_0_size)
    if cycle == 0:
        np.save(cfg.work_dir + '/X_L.npy', X_L)
        np.save(cfg.work_dir + '/X_U.npy', X_U)
        cfg = create_X_L_file(cfg, X_L, X_U, all_image_name)
    # ==================== cycle!=0 (train_partial_data)=====================================
    if cycle != 0:
        # load partial_labeled image
        ori_ann_file = modify_partial_data_set(cfg)
        cfg.data.train.label_type = 'partial'
        datasets = [build_dataset(cfg.data.train)]
        for data_info in datasets[0].data_infos:
            label_id_array = data_info["ann"]["labels"]
            for label_id in label_id_array:
                label_num[int(label_id)] += 1
        print("*" * 30)
        print(datasets)
        print("*" * 30)
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        print("**" * 30)
        print("Starting Training")
        print("**" * 30)
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.no_validate,
            timestamp=timestamp,
            meta=meta)

    # ==================== cycle!=0 (fully_labeled_dataset)=====================================
    if cycle != 0:
        X_L = np.load(cfg.work_dir + '/X_L.npy')
        X_U = np.load(cfg.work_dir + '/X_U.npy')
        # get the config of the labeled dataset
        cfg = create_X_L_file(cfg, X_L, X_U, all_image_name)
        cfg.data.train.ann_file = ori_ann_file
    cfg.data.train.label_type = 'full'
    datasets = [build_dataset(cfg.data.train)]
    for data_info in datasets[0].data_infos:
        label_id_array = data_info["ann"]["labels"]
        for label_id in label_id_array:
            label_num[int(label_id)] += 1
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    print("**" * 30)
    print("Starting Training")
    print("**" * 30)
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.no_validate,
        timestamp=timestamp,
        meta=meta)

    # ---------- Informative Image Selection ----------
    if cycle < 6:
        print("**"*30)
        print("Informative Image Selection")
        print("SAMPLE WAY::", args.al_sample)
        print("**" * 30)
        print("cycle::", cycle)
        cfg.data.val.load_type = 'select'
        dataset_al = build_dataset(cfg.data.val, dict(test_mode=True))
        data_loader = build_dataloader(dataset_al, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                       dist=False, shuffle=False)
        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed

        selectionmethod = SelectionMethod(args.al_sample, X_L, X_U, data_loader, model, cfg, all_image_name)
        selectionmethod.select(args.budget, args.iou_thresh, label_num,
                               args.score_thresh, args.class_num)
        # get the config of the labeled dataset
        cfg.work_dir = cfg.work_dir[:-1] + str(cycle+1)
        mmcv.mkdir_or_exist(cfg.work_dir)
        np.save(cfg.work_dir + '/X_L.npy', X_L)
        np.save(cfg.work_dir + '/X_U.npy', X_U)
        source_path = cfg.work_dir[:-6] + 'annfile/'
        target_path = cfg.work_dir + '/annfile/'
        shutil.copytree(source_path, target_path)


if __name__ == '__main__':
    main()
