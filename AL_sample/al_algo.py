import torch
import pickle
import copy
import numpy as np
from utils.utils import non_max_suppression_and_remove_overlap_with_gt
from models_da import create_grids
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import non_max_suppression, bbox_iou

"""
inf_out: [1, 10647, 25] (batch, anchor boxes, predictions)
train_out: 3-tuple, with shape:
[1, 3, 13, 13, 25]
[1, 3, 26, 26, 25]
[1, 3, 52, 52, 25]
"""

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image_c = image.to("cpu").numpy() / 255
    noise = np.random.normal(mean, var ** 0.5, image_c.shape)
    out = image_c + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # out = (out*255).long()
    return torch.from_numpy(out).float()


# def predict(model, im_data, im_info):
#     """
#     :param model: data yolo object
#     :param im_data: image tensor
#     :param im_info: fully supervised:1, partially supervised:0
#     """
#     with torch.no_grad():
#         output, tgt_output, src_od_loss, tgt_od_loss, DA_img_loss_cls1, \
#         DA_img_loss_cls2, DA_img_loss_cls3, tgt_DA_img_loss_cls1, \
#         tgt_DA_img_loss_cls2, tgt_DA_img_loss_cls3 = model(im_data, im_info, None, None, 0, 0,
#                                                            None, 1, None, None, 0, 0, False)
#     return output

def random_scoring_anchor_boxes(model_output):
    return [torch.rand(out.shape[:-1]) for out in model_output]


def our_scoring_indicator(model_output, ins_da_output, img_da_output,
                          pos_ins_weight=0.05, da_tradeoff=0.5):
    """

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(4)+conf+class
    :param ins_da_output: Tensor format: batch, predict(2), grid_x, grid_y
    :param pos_ins_weight: weight for positive instance. the importance for the neg instance.
    :return:
    """

    inconsistency, transferrability, final_scores = [], [], []
    scale_num = len(model_output)
    for i in range(scale_num):
        conf = torch.sigmoid(model_output[i][..., 4])
        # weight = torch.ones_like(conf)
        # weight[conf < 0.5] = pos_ins_weight
        unc_score = uncertainty_scoring_anchor_boxes(model_output[i])
        inconsistency.append((conf ** pos_ins_weight) * unc_score)
        img_trans = img_trans_score(img_da_output, normalize=False)
        ins_trans = da_scoring_anchor_boxes(ins_da_output[i])
        transferrability.append(ins_trans*img_trans)
        final_score = inconsistency[i] * transferrability[i].repeat(1, inconsistency[i].shape[1], 1, 1)
        final_scores.append(final_score)
    return final_scores


def our_scoring_anchor_boxes(model_output, ins_da_output, img_da_output,
                             pos_ins_weight=0.05, da_tradeoff=0.5):
    """

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(4)+conf+class
    :param ins_da_output: Tensor format: batch, predict(2), grid_x, grid_y
    :param pos_ins_weight: weight for positive instance. the importance for the neg instance.
    :return:
    """
    inconsistency, transferrability, final_scores = [], [], []
    scale_num = len(model_output)
    for i in range(scale_num):
        conf = torch.sigmoid(model_output[i][..., 4])
        # weight = torch.ones_like(conf)
        # weight[conf < 0.5] = pos_ins_weight
        unc_score = uncertainty_scoring_anchor_boxes(model_output[i])
        inconsistency.append((conf ** pos_ins_weight) * unc_score)
        img_trans = img_trans_score(img_da_output, normalize=False)
        ins_trans = da_scoring_anchor_boxes(ins_da_output[i])
        transferrability.append(ins_trans*img_trans)
        final_score = inconsistency[i] + da_tradeoff * transferrability[i].repeat(1, inconsistency[i].shape[1], 1, 1)
        final_scores.append(final_score)

        # # check types: results: all negative boxes
        # large_idx = (conf ** pos_ins_weight) * unc_score > 0.5
        # typs = conf > 0.5
        # types_sta = typs[large_idx]
        # total = types_sta.shape[0]
        # pos_c = torch.sum(types_sta)
        # neg_c = total - pos_c
        # print(f"pos: {pos_c}\tneg: {neg_c}\t\ttotal: {total}\tneg_ perc: {neg_c/total}")

    return inconsistency, transferrability, final_scores


def uncertainty_scoring_anchor_boxes(model_output):
    # solution 1
    bbxyp = model_output.shape
    unc = torch.zeros_like(model_output[..., 0])
    # calc bvsb for each anchor box and store the result in unc tensor
    conf = torch.sigmoid(model_output[..., 4])
    cls = torch.sigmoid(model_output[..., 5:])
    tp = torch.argsort(cls)

    # solution 1 dumb for loop
    # for i in range(bbxyp[2]):
    #     for j in range(bbxyp[3]):
    #         for k in range(bbxyp[1]):
    #             confi = conf[0, k, i, j]
    #             clsi = cls[0, k, i, j]
    #             tpi = tp[0, k, i, j]
    #             bvsbi = clsi[tpi[-1]] - clsi[tpi[-2]]
    #             # record
    #             unc[0, k, i, j] = abs(confi - bvsbi)
    #             # if True in torch.isnan(unc):
    #             #     print('nan, stop')

    # solution 2 Work!!
    scalars = torch.arange(bbxyp[2] * bbxyp[3] * bbxyp[1])*(bbxyp[4]-5)
    fst = tp[..., -1]
    # scd = tp[..., -2]
    scalars = scalars.reshape_as(fst).to(fst.device)
    fbv = torch.take(cls, scalars+fst)
    # sbv = torch.take(cls, scalars+scd)
    unc = (fbv - conf)**2

    # solution 3 Fail
    # unc = torch.zeros_like(model_output[..., 0])
    # conf = model_output[..., 4]
    # cls = model_output[..., 5:]
    # tp = torch.argsort(cls)
    # # tp2 = np.argsort(cls.cpu().detach().numpy())
    # bvsb = cls[..., tp[..., cls.shape[-1]-1]] - cls[..., tp[..., cls.shape[-1]-2]]  # fail
    # unc = (conf - bvsb) ** 2

    return unc


def uncertainty_LS_scoring_example(model, imgs, model_prediction,
                                   noise_level=(1e-4, 2e-4, 3e-4, 4e-4, 5e-4),
                                   device="cpu", conf_thres=0.5):
    """Implement Localization Stability in Arxiv19 - Localization-Aware Active Learning for Object Detection.
    Add noise to the image and evaluate the var of the output.

    get the detection results of the clear image.

    for noise_level_i in noise_level_N:
        get corr_box: the box with the highest IoU with all boxes that overlap ref_box
        calc Sb = IoU(ref_box, corr_box)/N
    \sigma_M{P(b)Sb}/\sigma_M{P(b)} (where P(b) is the highest class proba prediction for ref_box b, M boxes in total)
    """
    if model_prediction[0] is None:
        return 1
    # add noise
    # mo = model_prediction[0][:, 4] * model_prediction[0][:, 5:].max(1)
    high_prob_arr = dict()
    sbb = dict()
    for noise in noise_level:
        img_noise = gasuss_noise(image=imgs[0], mean=0, var=noise)
        img_noise = img_noise.unsqueeze(0).to(device)

        # predict
        domain_label = torch.zeros(imgs.shape[0]).to(device)
        with torch.no_grad():
            # Run model
            inf_out, train_out, da_out = model(img_noise, domain_label=domain_label)
            output = non_max_suppression(inf_out, conf_thres=max(conf_thres, 0.01))

        iou_arr = []
        # find box that correspond to each reference box
        # ref_box : (x1, y1, x2, y2, object_conf, class_conf, class)
        for it, ref_box in enumerate(model_prediction[0]):
            if output[0] is not None and len(output[0]) > 0:
                ious = bbox_iou(ref_box[0:4], output[0])
                max_iou = torch.max(ious)
            else:
                max_iou = 0

            # store values for calculating stablization
            high_proba_class = ref_box[5]
            high_prob_arr[it] = high_proba_class
            if it not in sbb.keys():
                sbb[it] = [max_iou]
            else:
                sbb[it].append(max_iou)

    # calc score
    norm_p = 0
    score = 0
    for k in sbb.keys():
        norm_p += high_prob_arr[k]
        sb = sum(sbb[k]) / len(noise_level)
        score += sb*high_prob_arr[k]

    return score / norm_p


def uncertainty_mean_conf_scorer(model_outputs, M=5):
    """Uncertainty in BAOD uncertainty evaluation method.
    Average entropy of each predicted box.

    Mean conf of the first M boxes!

    prediction format: (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    if model_outputs[0] is None:
        return 1
    mo = model_outputs[0][:, 4] * model_outputs[0][:, 5]
    if len(model_outputs) > M:
        sorted_conf = torch.argsort(mo)  # asend, small -> large
        model_outputs = model_outputs[0][sorted_conf[len(sorted_conf) - M:], :]
        mo = model_outputs[0][:, 4] * model_outputs[0][:, 5]

    return 1-torch.mean(mo)


def margin_avg_scorer(model_prediction, num_ins_class=None, const_numerator=1):
    """Brust et al. 18 AL 4 deep OD.
    average the margins of classification predictions of each instance with a certain weight.
    The weight is calculated by the number of instances of the specific class.
    wc = (instances + classes)/(instances_class + 1), which is inversely proportional to the instances_class.

    Parameters
    -----------
    num_ins_class: list
    number of instances of each class.
    """
    if model_prediction[0] is None:
        return 1
    # ref_box : (x1, y1, x2, y2, object_conf, class_conf, class)
    margin = 0
    for it, ref_box in enumerate(model_prediction[0]):
        class_vec = ref_box[5:]
        sorted_args = torch.argsort(class_vec)    # small to large
        margin += 1/(1 if num_ins_class is None else (num_ins_class[sorted_args[-1]]+1))*(class_vec[sorted_args[-1]]-class_vec[sorted_args[-2]])   # multiple the class weight
    avg_margin = margin/len(model_prediction[0])
    return avg_margin


def uncertain_map(train_out, scale=1, r=1, max_pooling_width=2):
    """Calc the uncertainty of an image from the pixel-level information
    ICCV'19 Active Learning for Deep Detection Neural Networks"""

    def entropy(one_dim_tensor):
        """Calc entrooy for each element in an array."""
        return -one_dim_tensor*torch.log(one_dim_tensor)-(1-one_dim_tensor)*torch.log(1-one_dim_tensor)

    # [batch, 3, 13, 13, 25]
    assert scale in list(range(len(train_out)))
    pred_map = train_out[scale]
    max_sigmoid_cls = torch.max(torch.sigmoid(pred_map[..., 5:]), dim=-1, keepdim=True)[0]
    # pred_map = None
    square_width = pred_map.shape[2]
    sijk = torch.zeros((3, square_width, square_width))
    for i in range(square_width):
        for j in range(square_width):
            start_i = max(0, i-r)
            start_j = max(0, j-r)
            end_i = min(i+r+1, square_width)
            end_j = min(j+r+1, square_width)
            # [3_siezes, row, col, cls_value]
            for size_type in range(3):
                pij_arr = max_sigmoid_cls[0, size_type, start_i:end_i, start_j:end_j, :].reshape(-1)
                sij = entropy(torch.mean(pij_arr)) - torch.mean(entropy(pij_arr))
                sijk[size_type][i][j] = sij
    sijk_sum = torch.sum(sijk, dim=0)
    # max pooling
    if max_pooling_width > 0 and square_width % (max_pooling_width) == 0:
        # stride the matrix
        new_shape = int(square_width / max_pooling_width)
        sijk_pooled = torch.zeros((new_shape, new_shape))
        # pooling_obj = nn.MaxPool1d(2, stride=2)
        # sijk_pooled = pooling_obj(sijk_sum)
        for row_id in np.arange(start=0, stop=square_width, step=max_pooling_width):
            for col_id in np.arange(start=0, stop=square_width, step=max_pooling_width):
                sijk_pooled[int(row_id/max_pooling_width),int(col_id/max_pooling_width)] = torch.max(sijk_sum[row_id:row_id+max_pooling_width, col_id:col_id+max_pooling_width])
        return torch.sum(sijk_pooled)
    else:
        return torch.sum(sijk_sum)


def least_confidence_scorer(model_outputs):
    """The uncertainty of an image is defined as the highest confidence among all predicted box.
    prediction format: (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    mo = model_outputs[0][:, 4] * model_outputs[0][:, 5]
    return 1 - torch.max(mo)




# def BAOD_sampler():
#     """BAOD uncertainty evaluation method"""
#     pass


def random_scorer(model_outputs=None):
    """randomly select a batch of example."""
    return np.random.uniform()

def unc_instance_scorer(model_output):
    """

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(4)+conf+class
    :param ins_da_output: Tensor format: batch, predict(2), grid_x, grid_y
    :param pos_ins_weight: weight for positive instance. the importance for the neg instance.
    :return:
    """
    inconsistency, transferrability, final_scores = [], [], []
    scale_num = len(model_output)
    for i in range(scale_num):
        conf = torch.sigmoid(model_output[i][..., 4])
        cla_max = torch.sigmoid(model_output[i][..., 5:].max(-1)[0])
        unc_score = 0.5 - torch.abs(conf * cla_max - 0.5)
        inconsistency.append(unc_score)
    return inconsistency


def AADA_scoring_example(trans_score, unc_score):
    """Active adversarial domain adaptation method."""
    return trans_score * unc_score


def get_candidate_boxes(model_output, output_scores, model, gt_boxes,
                        score_thres=0.1, nms_thres=0.5, img_size=416,
                        store_part=False, part_array=None, min_area=15):
    """Run NMS for anchor boxes with al scores and remove boxes overlaps with GT.
    Also filter out boxes whose width or height less than 10 px.

    :param model_output: Tensor format: batch, boxes_ind, grid_x, grid_y, boxes_coord(xywh)+conf+class
    :param output_scores: Tensor format: batch, boxes_ind, grid_x, grid_y, scores
    :param model: nn.Module OD model
    :param gt_boxes: torch.Tensor, shape: (n_boxes, (image_id class xywh))
    :param score_thres: threshold to filter out low score instances
    :param nms_thres: threshold to filter out overlapped boxes
    :return:
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    # compute scores for each box.
    # NMS filter out low scored & overlapped boxes
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    tp = []
    for i in range(len(model_output)):
        tp.append(model_output[i].clone().detach().cpu())
        # replace conf and cls scores with al score for NMS
        tp[i][..., 4] = output_scores[i]
        tp[i][..., 5:] = torch.ones(tp[i].shape[-1]-5)
        if store_part:
            assert part_array is not None
            tp[i][..., 5] = part_array[i]

        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            try:
                ng, anchor_vec, anchors = model.module.pure_yolo.module_list[model.yolo_layers[i]].ng, \
                                          model.module.pure_yolo.module_list[
                                              model.yolo_layers[i]].anchor_vec, model.module.pure_yolo.module_list[
                                              model.yolo_layers[i]].anchors
            except AttributeError:
                ng, anchor_vec, anchors = model.module.module_list[model.yolo_layers[i]].ng, model.module.module_list[
                    model.yolo_layers[i]].anchor_vec, model.module.module_list[model.yolo_layers[i]].anchors
        else:
            try:
                ng, anchor_vec, anchors = model.pure_yolo.module_list[model.yolo_layers[i]].ng, \
                                          model.pure_yolo.module_list[model.yolo_layers[i]].anchor_vec, \
                                          model.pure_yolo.module_list[model.yolo_layers[i]].anchors
            except AttributeError:
                ng, anchor_vec, anchors = model.module_list[model.yolo_layers[i]].ng, model.module_list[
                    model.yolo_layers[i]].anchor_vec, model.module_list[
                                              model.yolo_layers[i]].anchors
        # convert to NMS format
        # different nums in different scales.
        # calc vars
        nx, ny = ng.to("cpu")  # x and y grid size
        na = len(anchors)
        stride = img_size / max(ng)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid_xy = torch.stack((xv, yv), 2).view((1, 1, int(ny), int(nx), 2)).to("cpu")
        # build wh gains
        anchor_wh = anchor_vec.view(1, na, 1, 1, 2).to("cpu")

        tp[i][..., 0:2] = torch.sigmoid(tp[i][..., 0:2]) + grid_xy  # xy
        tp[i][..., 2:4] = torch.exp(tp[i][..., 2:4]) * anchor_wh  # wh yolo method
        # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
        tp[i][..., :4] *= stride
        # torch.sigmoid_(tp[i][..., 4:])    # no need to sigmoid again
        tp[i] = tp[i].view(1, -1, model_output[i].shape[-1])

    # concat tp
    pred_anchor_boxes = torch.cat(tp, dim=1)

    # filter low scored box
    # image_id class xywh
    if gt_boxes is not None:
        gt_boxes[:, 2:] *= img_size

    max_conf = torch.max(pred_anchor_boxes[0, :, 4])
    assert max_conf > 0
    if max_conf < score_thres:
        score_thres = torch.mean(pred_anchor_boxes[0, :, 4])

    return non_max_suppression_and_remove_overlap_with_gt(pred_anchor_boxes, gt_boxes, conf_thres=score_thres,
                                                          nms_thres=nms_thres, min_area=min_area)