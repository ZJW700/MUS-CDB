import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from mmdet.utils import get_root_logger
from mmdet.apis import show_result_pyplot, single_gpu_test
from mmdet.apis import inference_detector
from tools.AL_utils.compute_budget import *
import cv2
from mmdet.datasets.pipelines import Compose
from mmrotate.core import obb2xyxy


def IGUS_select(X_U, budget, iou_thresh, data_loader, model, cfg, score_thresh):
    """Uncertainty in BAOD uncertainty evaluation method.
          Average entropy of each predicted box.
          or
          Mean conf of the first M boxes.
   """
    model.eval()
    model.cuda(cfg.gpu_ids[0])  # 将模型加载到GPU上去
    all_bbox_info = torch.Tensor().cuda(cfg.gpu_ids[0])
    partial_queried_img = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if i in X_U:
                # 获得该值的真实编号
                idx = np.where(i == X_U)[0][0]
                (shotname, ext) = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])
                partial_queried_img.append(shotname)
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _ = model(return_loss=False, return_entropy=True, **data)
                valid_mask_thr = det_bboxes[:, -1] > score_thresh
                inds_thr = valid_mask_thr.nonzero(as_tuple=False).squeeze(1)
                det_bboxes_thr = det_bboxes[inds_thr]
                if len(entropys) > 0:
                    score = det_bboxes_thr[:, -1]
                    mean_score = score.mean()
                    weight_score = 1 - mean_score
                    score = weight_score*entropys.cuda(cfg.gpu_ids[0])
                    det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                    image_id_2D = torch.Tensor([idx]).repeat(len(det_bboxes), 1).cuda(cfg.gpu_ids[0])
                    bbox_info = torch.cat((det_bboxes, det_labels_2D, score, image_id_2D),
                                          1)  # (cx,cy,w,h,a,score,label,entropy,image_idx)
                    all_bbox_info = torch.cat((all_bbox_info, bbox_info), dim=0)
            if i % 500 == 0:
                print(f'------ {i}/{len(data_loader.dataset)} ------')
    index = (all_bbox_info[:, -2]).argsort()  # asend, small -> large
    new_all_bbox_info = all_bbox_info[index].cpu()
    torch.save(new_all_bbox_info,
               cfg.work_dir + '/all_candidate_bbox_info.pt')
    annotate_budget_5000(new_all_bbox_info, partial_queried_img, cfg.work_dir[:-6] + 'annfile/', budget, iou_thresh)


def DGCB_select(X_U, budget, iou_thresh, data_loader, model, cfg, label_num, class_num):
    """Uncertainty in BAOD uncertainty evaluation method.
          Average entropy of each predicted box.
          or
          Mean conf of the first M boxes.
   """
    model.eval()
    model.cuda(cfg.gpu_ids[0])  # 将模型加载到GPU上去
    all_bbox_info = torch.Tensor().cuda(cfg.gpu_ids[0])
    partial_queried_img = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if i in X_U:
                # 获得该值的真实编号
                idx = np.where(i == X_U)[0][0]
                (shotname, ext) = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])
                partial_queried_img.append(shotname)
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _ = model(return_loss=False, return_entropy=True, **data)
                if len(entropys) > 0:
                    score = entropys.cuda(cfg.gpu_ids[0])
                    det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])  # 一维变二维(n,1)
                    image_id_2D = torch.Tensor([idx]).repeat(len(det_bboxes), 1).cuda(cfg.gpu_ids[0])  # 同上     (n,1)
                    bbox_info = torch.cat((det_bboxes, det_labels_2D, score, image_id_2D),
                                          1)  # (cx,cy,w,h,a,score,label,entropy,image_idx)
                    all_bbox_info = torch.cat((all_bbox_info, bbox_info), dim=0)
            if i % 500 == 0:
                print(f'------ {i}/{len(data_loader.dataset)} ------')
    index = (all_bbox_info[:, -2]).argsort()  # asend, small -> large
    new_all_bbox_info = all_bbox_info[index].cpu()
    torch.save(new_all_bbox_info,
               cfg.work_dir + '/all_candidate_bbox_info.pt')
    annotate_budget_5000_class(new_all_bbox_info, partial_queried_img, cfg.work_dir[:-6] + 'annfile/',
                                   budget, iou_thresh, label_num, class_num)


def mus_cdb_select(X_U, budget, iou_thresh, data_loader, model, cfg, label_num, class_num, score_thresh):
    """Uncertainty in BAOD uncertainty evaluation method.
          Average entropy of each predicted box.
          or
          Mean conf of the first M boxes.
   """
    model.eval()
    model.cuda(cfg.gpu_ids[0])
    all_bbox_info = torch.Tensor().cuda(cfg.gpu_ids[0])
    partial_queried_img = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if i in X_U:
                idx = np.where(i == X_U)[0][0]
                (shotname, ext) = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])
                partial_queried_img.append(shotname)
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _ = model(return_loss=False, return_entropy=True, **data)
                valid_mask_thr = det_bboxes[:, -1] > score_thresh
                inds_thr = valid_mask_thr.nonzero(as_tuple=False).squeeze(1)
                det_bboxes_thr = det_bboxes[inds_thr]
                if len(entropys) > 0:
                    score = det_bboxes_thr[:, -1]
                    mean_score = score.mean()
                    weight_score = 1 - mean_score
                    score = weight_score * entropys.cuda(cfg.gpu_ids[0])
                    det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                    image_id_2D = torch.Tensor([idx]).repeat(len(det_bboxes), 1).cuda(cfg.gpu_ids[0])
                    bbox_info = torch.cat((det_bboxes, det_labels_2D, score, image_id_2D),
                                          1)  # (cx,cy,w,h,a,score,label,entropy,image_idx)
                    all_bbox_info = torch.cat((all_bbox_info, bbox_info), dim=0)
            if i % 500 == 0:
                print(f'------ {i}/{len(data_loader.dataset)} ------')
    index = (all_bbox_info[:, -2]).argsort()  # asend, small -> large
    new_all_bbox_info = all_bbox_info[index].cpu()
    torch.save(new_all_bbox_info,
               cfg.work_dir + '/all_candidate_bbox_info.pt')
    annotate_budget_5000_class(new_all_bbox_info, partial_queried_img, cfg.work_dir[:-6] + 'annfile/',
                                           budget, iou_thresh, label_num, class_num)


class SelectionMethod:
    """
    Abstract base class for selection methods,
    which allow to select a subset of indices from the pool set as the next batch to label for Batch Active Learning.
    """
    def __init__(self, al_sample, X_L, X_U, data_loader, model, cfg, all_image_name):
        super().__init__()
        self.logger = get_root_logger()
        self.al_sample = al_sample
        self.X_L = X_L
        self.X_U = X_U
        self.data_loader = data_loader
        self.model = model
        self.cfg = cfg
        self.all_image_name = all_image_name

    def select(self, budget, iou_thresh, label_num=0, score_thresh=0.05, class_num=15):
        """
        Select selection_size elements from the pool set
        (which is assumed to be given in the constructor of the corresponding subclass).
        This method needs to be implemented by subclasses.
        It is assumed that this method is only called once per object, since it can modify the state of the object.

        Args:
            selection_size (int): how much images selected in one cycle

        Returns:
            idxs_selected (np.ndarray): index of chosen images
        """
        if self.al_sample == 'IGUS':
            self.logger.info(f'------ IGUS ------')
            return IGUS_select(self.X_U, budget, iou_thresh, self.data_loader, self.model,
                                            self.cfg, score_thresh)
        elif self.al_sample == 'DGCB':
            self.logger.info(f'------ DGCB ------')
            return DGCB_select(self.X_U, budget, iou_thresh, self.data_loader, self.model,
                                              self.cfg, label_num, class_num)
        elif self.al_sample == 'mus-cdb':
            self.logger.info(f'------ mus-cdb ------')
            return mus_cdb_select(self.X_U, budget, iou_thresh, self.data_loader, self.model,
                                        self.cfg, label_num, class_num, score_thresh)