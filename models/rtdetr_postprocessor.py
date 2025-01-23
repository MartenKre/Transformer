"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from . import box_ops

__all__ = ['RTDETRPostProcessor']

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
