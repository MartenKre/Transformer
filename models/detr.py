# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from json import encoder
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import angle, nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .position_encoding import pos_encode_zoom
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, input_dim_gt, aux_loss=False, use_embeddings=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.objectness_embed = nn.Linear(hidden_dim, 1) # only one output class -> objectness
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if not use_embeddings:
            self.query_embed = MLP(input_dim_gt, hidden_dim//2, hidden_dim, 3) # embedding: (dist,bearing) -> embedding
        else:
            self.query_embed_1 = nn.Embedding(40, int(hidden_dim/input_dim_gt))
            self.query_embed_2 = nn.Embedding(80, int(hidden_dim/input_dim_gt))
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        # test
        self.use_embeddings = use_embeddings

    def forward(self, images, queries, queries_mask):
        """ The forward expects a NestedTensor, which consists of:
               - images.tensor: batched images, of shape [batch_size x 3 x H x W]
               - images.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - buoyData: gt data of shape [batch_size x 3] -> dist, angle1, angle2

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos = self.backbone(images)


        encoder_embed = self.input_proj(features)

        if self.use_embeddings:
            dist_input = (queries[...,0].clamp(min=0,max=1) * 1000) // 25
            angle_input = ((queries[...,1].clamp(min=-1, max=1)+1) * 500) // 12.5
            decoder_embed = torch.cat((self.query_embed_1(dist_input.int()), self.query_embed_2(angle_input.int())), dim = -1) # [N, Seq_len, 256] query embedding 
        else:
            decoder_embed = self.query_embed(queries)


        hs = self.transformer(features, encoder_embed, decoder_embed, pos, queries_mask) # returns [Num_Decoding, Batch_SZ, Seq_len, hidden_dim]

        outputs_objectness = self.objectness_embed(hs).sigmoid().squeeze(dim=-1) # [Num_Decoding, N, Seq_len]
        outputs_coord = self.bbox_embed(hs).sigmoid() # [Num_Decoding, N, Seq_len, 4]

        out = {'pred_logits': outputs_objectness[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_objectness, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_l)
        num_l = torch.clamp(num_l / get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_q = queries_mask.sum()
        if is_dist_avail_and_initialized():
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


class PostProcess(nn.Module):
    """ This module converts the BB tensor from cxcywh to xyxy w.r.t original image size (not used during training"""
    @torch.no_grad()
    def forward(self, outputs, target_size=[1080, 1920]):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_size: List of Image Size H, W
        """

        out_logits, out_bbox = outputs['pred_logits'].squeeze(0), outputs['pred_boxes'].squeeze(0)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device).repeat(out_bbox.size(dim=0), 1)
        boxes = boxes * scale_fct

        results = {'objectness': out_logits, 'boxes': boxes}

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
