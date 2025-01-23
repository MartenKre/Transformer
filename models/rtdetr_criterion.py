"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


from . import box_ops

from .dist import get_world_size, is_dist_available_and_initialized


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses


    def loss_labels(self, outputs, labels, queries_mask, labels_mask, num_q, num_l):
        """Classification loss (NLL)
        outputs: dict containing pred_logits key that points to objectness pred tensor
        targets: tensor with size (B_SZ, Max_seq_len, 5)
        queries_mask: for each sample in batch contains binary mask for each outputs
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        targets = torch.zeros_like(src_logits) 
        targets[labels_mask] = 1.0 # target mask to find labels idx where objectness = 1

        loss_bce = F.binary_cross_entropy(src_logits, targets, reduction='none')
        loss_bce = loss_bce[queries_mask].sum() /  num_q     # compute mean loss (only for non padded elements)
        losses = {'loss_bce': loss_bce}

        return losses


    def loss_boxes(self, outputs, labels, queries_mask, labels_mask, num_q, num_l):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            outputs: dict containing pred_boxess key that points to boxes pred tensor
            targets: tensor with size (B_SZ, Max_seq_len, 5)
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']

        target_boxes = labels[..., 1:]  # only extract relevant BB labels (not query id)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox[labels_mask].sum() / num_l # filter loss based on elements than contain gt label

        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes).flatten(start_dim=0, end_dim=1)
        target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes).flatten(0, 1)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_xyxy,target_xyxy))
        losses['loss_giou'] = loss_giou[labels_mask.flatten(0,1)].sum() / num_l

        return losses

    def get_loss(self, loss, outputs, labels, queries_mask, labels_mask, num_q, num_l):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, labels, queries_mask, labels_mask, num_q, num_l)

    def forward(self, outputs, labels, queries_mask, labels_mask):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_l = labels_mask.sum()
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_l)
        num_l = torch.clamp(num_l / get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_q = queries_mask.sum()
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_q)
        num_q = torch.clamp(num_q / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, labels, queries_mask, labels_mask, num_q, num_l))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, labels, queries_mask, labels_mask, num_q, num_l)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



