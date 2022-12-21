import numpy as np
import os
import torch
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmcv.ops import box_iou_rotated
import torch.nn.functional as F

CLASSES = ('plane', 'baseball-diamond',
           'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle',
           'ship', 'tennis-court',
           'basketball-court', 'storage-tank',
           'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool',
           'helicopter', 'container-crane',
           'airport', 'helipad')


def annotate_budget_5000(all_bbox_info, partial_queried_img, data_root=' ', budget_size=5000, iou_thresh=0):
    budget = 0
    for candidate_box in all_bbox_info:
        image_idx = int(candidate_box[-1])
        img_name = partial_queried_img[image_idx]
        det_bbox = candidate_box[:6]
        # Fliter out the candidate bboxes which are highly overlapped with queried set.(save IOU<0.5)
        gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'queried', data_root)
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
            if max_iou >= 0.5:
                continue
        gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'unqueried', data_root)
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
            if max_iou > iou_thresh:
                queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
            else:
                queried_gt = []
                unqueried_gt = gt_raw[(ious <= iou_thresh).cpu().detach().numpy()]
            if len(queried_gt) > 0:
                for i in range(len(queried_gt)):
                    budget += 1
                updata_image_info(queried_gt, unqueried_gt, img_name, data_root)
        if budget == budget_size:
            break
    form_file(data_root)


def annotate_budget_5000_class(all_bbox_info, partial_queried_img, data_root=' ', budget_size=5000, iou_thresh=0, label_num=0, class_num=15):
    label_sum = label_num.sum()
    label_ratio = (1 - label_num / label_sum)
    label_num = F.softmax(label_ratio) * budget_size

    all_bbox_info_rest = []
    budget_class = torch.zeros(class_num+1)
    for candidate_box in all_bbox_info:
        image_idx = int(candidate_box[-1])
        img_name = partial_queried_img[image_idx]
        label_id = int(candidate_box[6])
        if budget_class[label_id] >= label_num[label_id]:
            all_bbox_info_rest.append(candidate_box.numpy())
            continue
        det_bbox = candidate_box[:6]
        # Fliter out the candidate bboxes which are highly overlapped with queried set.(save IOU<0.5)
        gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'queried', data_root)
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
            if max_iou >= 0.5:
                continue
        gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'unqueried', data_root)
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
            if max_iou > iou_thresh:
                queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
            else:
                queried_gt = []
                unqueried_gt = gt_raw[(ious <= iou_thresh).cpu().detach().numpy()]
            if len(queried_gt) > 0:
                for i in range(len(queried_gt)):
                    budget_class[label_id] += 1
                    budget_class[class_num] += 1
                updata_image_info(queried_gt, unqueried_gt, img_name, data_root)
        if budget_class[class_num] == budget_size:
            break
    if budget_class[class_num] < budget_size:
        for candidate_box in all_bbox_info_rest:
            image_idx = int(candidate_box[-1])
            img_name = partial_queried_img[image_idx]
            det_bbox = candidate_box[:6]
            # Fliter out the candidate bboxes which are highly overlapped with queried set.(save IOU<0.5)
            gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'queried', data_root)
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
                if max_iou >= 0.5:
                    continue
            gt_bboxes, gt_labels, gt_polygons, gt_raw = get_gt_bboxes(img_name, 'unqueried', data_root)
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                max_iou, ious = calc_iou(torch.Tensor(det_bbox[:5]), gt_bboxes)
                if max_iou > iou_thresh:
                    queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                    unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                else:
                    queried_gt = []
                    unqueried_gt = gt_raw[(ious <= iou_thresh).cpu().detach().numpy()]
                if len(queried_gt) > 0:
                    for i in range(len(queried_gt)):
                        budget_class[class_num] += 1
                    updata_image_info(queried_gt, unqueried_gt, img_name, data_root)
            if budget_class[class_num] == budget_size:
                break
    form_file(data_root)


def cal_instance_num(idx, all_image_name, data_root=' '):
    name = all_image_name[idx]
    instance_num = len(open(data_root + name + '.txt').readlines())
    return instance_num


def get_gt_bboxes(img_name, way, data_root):
    """
       Removes detections with lower object confidence score than 'conf_thres'
       Non-Maximum Suppression to further filter detections.
       Returns gt_bboxes with shape:
           (cx,cy,w,h,a,class)
    """

    cls_map = {c: i
               for i, c in enumerate(CLASSES)
               }
    cls_map.update({'background': -1})

    path = data_root + way + '/' + img_name + ".txt"

    gt_bboxes = []
    gt_labels = []
    gt_polygons = []
    if not os.path.exists(path):
        print(path+"不存在，先创建")
        os.system(r"touch {}".format(path))
    with open(path, 'r') as f:
        s = f.readlines()
        new_s = []
        if len(s) != 0:
            for si in s:
                bbox_info = si.split()
                poly = np.array(bbox_info[:8], dtype=np.float32)
                try:
                    x, y, w, h, a = poly2obb_np(poly, "le90")
                except:  # noqa: E722
                    continue
                cls_name = bbox_info[8]
                difficulty = int(bbox_info[9])
                label = cls_map[cls_name]
                if difficulty > 100:
                    pass
                else:
                    gt_bboxes.append([x, y, w, h, a])
                    gt_labels.append(label)
                    gt_polygons.append(poly)
                    new_s.append(si)
    return torch.Tensor(gt_bboxes), gt_labels, gt_polygons, np.array(new_s)


def updata_image_info(queried_gt, unqueried_gt, img_name, data_root):
    queried_pth = data_root + "queried/" + img_name + ".txt"
    unqueried_path = data_root + "unqueried/" + img_name + ".txt"
    if not os.path.exists(queried_pth):
        print(queried_pth+"不存在，先创建")
        os.system(r"touch {}".format(queried_pth))
    with open(queried_pth, 'a') as f1:
        for i in queried_gt:
            f1.write(i)
    with open(unqueried_path, 'w+') as f2:
        for i in unqueried_gt:
            f2.write(i)


def updata_imgae_background(background_proposal, img_name, data_root):
    info = ''
    for i in range(len(background_proposal)):
        info = info + str(background_proposal[i]) + " "
    info += 'background  0' + str('\n')

    queried_pth = data_root + "queried/" + img_name + ".txt"
    if not os.path.exists(queried_pth):
        print(queried_pth+"不存在，先创建")
        os.system(r"touch {}".format(queried_pth))
    with open(queried_pth, 'a') as f1:
        f1.write(info)


def calc_iou(bbox1, bboxes2):
    assert (bbox1.size(-1) == 5 or bbox1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    bboxes1 = bbox1.repeat(len(bboxes2), 1)
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    ious = box_iou_rotated(clamped_bboxes1, clamped_bboxes2, 'iou', True)
    max_iou = ious.max().item()

    return max_iou, ious


def form_file(data_root):
    dir = data_root + 'queried/'
    lis = os.listdir(dir)
    txt_path = data_root + 'trainval_X_L.txt'
    write_txt(lis, txt_path)


def write_txt(lis, path):
    f = open(path, 'w+')
    for i in range(0, len(lis)):
        (shotname, ext) = os.path.splitext(lis[i])
        f.write(shotname + '\n')
    f.close()