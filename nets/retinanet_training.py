import math
from functools import partial

import torch
import torch.nn as nn


def calc_iou(a, b):
    max_length = torch.max(a)
    a = a / max_length
    b = b / max_length
    
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def get_target(anchor, bbox_annotation, classification, cuda):
    #------------------------------------------------------#
    #   计算真实框和先验框的交并比
    #   anchor              num_anchors, 4
    #   bbox_annotation     num_true_boxes, 5
    #   Iou                 num_anchors, num_true_boxes
    #------------------------------------------------------#
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
    
    #------------------------------------------------------#
    #   计算与先验框重合度最大的真实框
    #   IoU_max             num_anchors,
    #   IoU_argmax          num_anchors,
    #------------------------------------------------------#
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)

    #------------------------------------------------------#
    #   寻找哪些先验框在计算loss的时候需要忽略
    #------------------------------------------------------#
    targets = torch.ones_like(classification) * -1
    targets = targets.type_as(classification)

    #------------------------------------------#
    #   重合度小于0.4需要参与训练
    #------------------------------------------#
    targets[torch.lt(IoU_max, 0.4), :] = 0

    #--------------------------------------------------#
    #   重合度大于0.5需要参与训练，还需要计算回归loss
    #--------------------------------------------------#
    positive_indices = torch.ge(IoU_max, 0.5)

    #--------------------------------------------------#
    #   取出每个先验框最对应的真实框
    #--------------------------------------------------#
    assigned_annotations = bbox_annotation[IoU_argmax, :]

    #--------------------------------------------------#
    #   将对应的种类置为1
    #--------------------------------------------------#
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    #--------------------------------------------------#
    #   计算正样本数量
    #--------------------------------------------------#
    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations

def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    #--------------------------------------------------#
    #   取出作为正样本的先验框对应的真实框
    #--------------------------------------------------#
    assigned_annotations = assigned_annotations[positive_indices, :]

    #--------------------------------------------------#
    #   取出作为正样本的先验框
    #--------------------------------------------------#
    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    #--------------------------------------------------#
    #   计算真实框的宽高与中心
    #--------------------------------------------------#
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    #---------------------------------------------------#
    #   利用真实框和先验框进行编码，获得应该有的预测结果
    #---------------------------------------------------#
    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()
    return targets

def calculate_giou_loss(pred_boxes, gt_boxes):
    """计算GIoU损失
    
    Args:
        pred_boxes: 预测边界框坐标 [x1, y1, x2, y2]
        gt_boxes: 真实边界框坐标 [x1, y1, x2, y2]
    
    Returns:
        GIoU损失值
    """
    # 确保边界框格式为 [x1, y1, x2, y2]
    assert pred_boxes.shape[1] == 4
    assert gt_boxes.shape[1] == 4
    
    # 确保坐标顺序正确，让x1 <= x2, y1 <= y2
    pred_x1 = torch.min(pred_boxes[:, 0], pred_boxes[:, 2])
    pred_y1 = torch.min(pred_boxes[:, 1], pred_boxes[:, 3])
    pred_x2 = torch.max(pred_boxes[:, 0], pred_boxes[:, 2])
    pred_y2 = torch.max(pred_boxes[:, 1], pred_boxes[:, 3])
    
    gt_x1 = torch.min(gt_boxes[:, 0], gt_boxes[:, 2])
    gt_y1 = torch.min(gt_boxes[:, 1], gt_boxes[:, 3])
    gt_x2 = torch.max(gt_boxes[:, 0], gt_boxes[:, 2])
    gt_y2 = torch.max(gt_boxes[:, 1], gt_boxes[:, 3])
    
    # 计算交集的坐标
    x1_inter = torch.max(pred_x1, gt_x1)
    y1_inter = torch.max(pred_y1, gt_y1)
    x2_inter = torch.min(pred_x2, gt_x2)
    y2_inter = torch.min(pred_y2, gt_y2)

    # 计算交集面积，确保宽高非负
    w_inter = torch.clamp(x2_inter - x1_inter, min=0)
    h_inter = torch.clamp(y2_inter - y1_inter, min=0)
    area_inter = w_inter * h_inter

    # 计算预测框和真实框的面积，确保面积为正
    area_pred = torch.clamp((pred_x2 - pred_x1) * (pred_y2 - pred_y1), min=1e-6)
    area_gt = torch.clamp((gt_x2 - gt_x1) * (gt_y2 - gt_y1), min=1e-6)
    
    # 计算并集面积
    area_union = area_pred + area_gt - area_inter

    # 计算IoU，添加数值稳定性
    iou = area_inter / (area_union + 1e-7)

    # 计算最小闭包区域的坐标
    x1_c = torch.min(pred_x1, gt_x1)
    y1_c = torch.min(pred_y1, gt_y1)
    x2_c = torch.max(pred_x2, gt_x2)
    y2_c = torch.max(pred_y2, gt_y2)

    # 计算闭包区域面积，确保非零
    w_c = torch.max(x2_c - x1_c, torch.ones_like(x2_c) * 1e-6)
    h_c = torch.max(y2_c - y1_c, torch.ones_like(y2_c) * 1e-6)
    area_c = w_c * h_c

    # 计算GIoU
    giou = iou - (area_c - area_union) / (area_c + 1e-7)
    
    # 确保giou值在合理范围内，防止极端值
    giou = torch.clamp(giou, min=-1.0, max=1.0)

    # 计算GIoU损失 (1 - GIoU)
    loss = 1.0 - giou

    return loss


def decode_bbox(regression, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    """将预测的偏移量解码为边界框坐标"""
    # 获取预测的偏移量，注意encode_bbox中的顺序是(dy, dx, dh, dw)
    ty = regression[:, 0]  # y方向偏移
    tx = regression[:, 1]  # x方向偏移
    th = regression[:, 2]  # 高度缩放
    tw = regression[:, 3]  # 宽度缩放

    # 获取对应锚框的属性
    widths = anchor_widths[positive_indices]
    heights = anchor_heights[positive_indices]
    ctr_x = anchor_ctr_x[positive_indices]
    ctr_y = anchor_ctr_y[positive_indices]

    # 解码边界框中心坐标和宽高
    pred_ctr_x = tx * widths + ctr_x
    pred_ctr_y = ty * heights + ctr_y
    pred_w = torch.exp(tw) * widths
    pred_h = torch.exp(th) * heights

    # 转换为[x1, y1, x2, y2]格式
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    # 组合边界框坐标
    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

    return pred_boxes



class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha = 0.25, gamma = 2.0, cuda = True, epoch = 0):
        #---------------------------#
        #   获得batch_size的大小
        #---------------------------#
        batch_size = classifications.shape[0]

        #--------------------------------------------#
        #   获得先验框，将先验框转换成中心宽高的形式
        #--------------------------------------------#
        dtype = regressions.dtype
        anchor = anchors[0, :, :].to(dtype)
        #--------------------------------------------#
        #   将先验框转换成中心，宽高的形式
        #--------------------------------------------#
        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []
        for j in range(batch_size):
            #-------------------------------------------------------#
            #   取出每张图片对应的真实框、种类预测结果和回归预测结果
            #-------------------------------------------------------#
            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            
            classification = torch.clamp(classification, 5e-4, 1.0 - 5e-4)
            
            if len(bbox_annotation) == 0:
                #-------------------------------------------------------#
                #   当图片中不存在真实框的时候，所有特征点均为负样本
                #-------------------------------------------------------#
                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = alpha_factor.type_as(classification)

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                #-------------------------------------------------------#
                #   计算特征点对应的交叉熵
                #-------------------------------------------------------#
                bce = - (torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                classification_losses.append(cls_loss.sum())
                #-------------------------------------------------------#
                #   回归损失此时为0
                #-------------------------------------------------------#
                regression_losses.append(torch.tensor(0).type_as(classification))
                    
                continue

            #------------------------------------------------------#
            #   计算真实框和先验框的交并比
            #   targets                 num_anchors, num_classes
            #   num_positive_anchors    正样本的数量
            #   positive_indices        num_anchors, 
            #   assigned_annotations    num_anchors, 5
            #------------------------------------------------------#
            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, 
                                                                                        bbox_annotation, classification, cuda)
            
            #------------------------------------------------------#
            #   首先计算交叉熵loss
            #------------------------------------------------------#
            alpha_factor = torch.ones_like(targets) * alpha
            alpha_factor = alpha_factor.type_as(classification)
            #------------------------------------------------------#
            #   这里使用的是Focal loss的思想，
            #   易分类样本权值小
            #   难分类样本权值大
            #------------------------------------------------------#
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            #------------------------------------------------------#
            #   把忽略的先验框的loss置为0
            #------------------------------------------------------#
            zeros = torch.zeros_like(cls_loss)
            zeros = zeros.type_as(cls_loss)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            
            #------------------------------------------------------#
            #   如果存在先验框为正样本的话
            #------------------------------------------------------#
            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
                
                # 解码预测偏移量为实际边界框坐标
                pred_boxes = decode_bbox(regression[positive_indices, :], positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
                
                # 获取对应的真实边界框坐标
                gt_boxes = assigned_annotations[positive_indices, :4]
                
                # 确保边界框坐标有效
                pred_boxes = torch.clamp(pred_boxes, min=0)
                gt_boxes = torch.clamp(gt_boxes, min=0)
                
                # 确保边界框排序正确(x1<x2, y1<y2) - 使用向量化操作
                # 对预测框
                x1_pred = torch.min(pred_boxes[:, 0], pred_boxes[:, 2])
                y1_pred = torch.min(pred_boxes[:, 1], pred_boxes[:, 3])
                x2_pred = torch.max(pred_boxes[:, 0], pred_boxes[:, 2])
                y2_pred = torch.max(pred_boxes[:, 1], pred_boxes[:, 3])
                pred_boxes = torch.stack([x1_pred, y1_pred, x2_pred, y2_pred], dim=1)
                
                # 对真实框
                x1_gt = torch.min(gt_boxes[:, 0], gt_boxes[:, 2])
                y1_gt = torch.min(gt_boxes[:, 1], gt_boxes[:, 3])
                x2_gt = torch.max(gt_boxes[:, 0], gt_boxes[:, 2])
                y2_gt = torch.max(gt_boxes[:, 1], gt_boxes[:, 3])
                gt_boxes = torch.stack([x1_gt, y1_gt, x2_gt, y2_gt], dim=1)
                
                # 计算GIoU损失
                giou_loss = calculate_giou_loss(pred_boxes, gt_boxes)
                
                # 直接使用GIoU损失作为回归损失
                regression_loss = giou_loss.mean()
                
                regression_losses.append(regression_loss)
            else:
                regression_losses.append(torch.tensor(0).type_as(classification))
        
        # 计算平均loss并返回
        c_loss = torch.stack(classification_losses).mean()
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
