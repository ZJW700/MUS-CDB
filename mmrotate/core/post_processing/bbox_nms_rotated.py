# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_rotated
from mmcv.ops import box_iou_rotated


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           stage1_rois,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)                DOTA:(n, #class*5)  HRSC:(n, 5)   n=2000
        multi_scores (torch.Tensor): shape (n, #class), where the last column     n:预测框数量(2000)
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1   # DOTA:15  (0,1,2...,15)  15:backkground   HRSC:1-->(0,1)  0:bckground  1:ship
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    # ===================== added:compute bg score ====================
    if multi_bboxes.shape[1] > 5:
        bg_scores = multi_scores[:, -1]
        bg_scores_15 = [bg_scores[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(bg_scores.size(0))]  # bg_score分配给每个bbox(*15)    # multi_bboxes.shape[1]/5 = 15(DOTA)  33(HRSC)
        bg_scores_15_new = torch.Tensor(
            [item.cpu().detach().numpy() for item in bg_scores_15])  # 由len=2000(2000,15)的tensor_list变为一个tensor
        bg_scores = bg_scores_15_new.view(-1, 1)                    # torch.Size([30000, 1])   (30000=2000*15)
    else:
        bg_scores = multi_scores[:, -1]
    # ===================== added:compute entropy =====================\
    if multi_bboxes.shape[1] > 5:
        probs = scores
        log_probs = torch.log(probs)
        entropys = (probs * log_probs).sum(1)    # 熵的负值
        entropys_15 = [entropys[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(entropys.size(0))]  # entropy分配给每个bbox(*15)
        entropys_15_new = torch.Tensor(
            [item.cpu().detach().numpy() for item in entropys_15])  # 由len=2000(2000,15)的tensor_list变为一个tensor
        entropys = entropys_15_new.view(-1, 1)                      # torch.Size([30000, 1])   (30000=2000*15)
    else:
        probs = scores
        log_probs = torch.log(probs)
        entropys = (probs * log_probs).sum(1)  # 熵的负值


    # ==================================================================
    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # ===================== added:cal iou =============================
    # if multi_bboxes.shape[1] > 5:
    #     stage1_rois_15 = [stage1_rois[i].repeat(15) for i in range(stage1_rois.size(0))]  # stage1_rois分配给每个bbox(*15)
    #     stage1_rois_15_new = torch.Tensor(  # torch.Size([2000, 75])
    #         [item.cpu().detach().numpy() for item in stage1_rois_15])
    #     stage1_rois = stage1_rois_15_new.view(-1, 5)
    #     bboxes_zjw = bboxes.cpu()
    #     zjw_ious = box_iou_rotated(stage1_rois, bboxes_zjw, 'iou', True)
    #     zjw_ious = zjw_ious.unsqueeze(-1).cpu()
    # else:
    # ===================== added:compute bg score ====================

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    entropys = entropys[inds]    # added
    bg_scores = bg_scores[inds]  # added
    # zjw_ious = zjw_ious[inds]    # added

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels, entropys, bg_scores

    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    entropys = entropys[keep]    # added
    bg_scores = bg_scores[keep]  # added
    # zjw_ious = zjw_ious[keep]    # added

    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels, entropys, bg_scores


def aug_multiclass_nms_rotated(merged_bboxes, merged_labels, score_thr, nms,
                               max_num, classes):
    """NMS for aug multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        classes (int): number of classes.

    Returns:
        tuple (dets, labels): tensors of shape (k, 5), and (k). Dets are boxes
            with scores. Labels are 0-based.
    """
    bboxes, labels = [], []

    for cls in range(classes):
        cls_bboxes = merged_bboxes[merged_labels == cls]
        inds = cls_bboxes[:, -1] > score_thr
        if len(inds) == 0:
            continue
        cur_bboxes = cls_bboxes[inds, :]
        cls_dets, _ = nms_rotated(cur_bboxes[:, :5], cur_bboxes[:, -1],
                                  nms.iou_thr)
        cls_labels = merged_bboxes.new_full((cls_dets.shape[0], ),
                                            cls,
                                            dtype=torch.long)
        if cls_dets.size()[0] == 0:
            continue
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, _inds = bboxes[:, -1].sort(descending=True)
            _inds = _inds[:max_num]
            bboxes = bboxes[_inds]
            labels = labels[_inds]
    else:
        bboxes = merged_bboxes.new_zeros((0, merged_bboxes.size(-1)))
        labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
