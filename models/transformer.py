# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Value, device, nn, Tensor
from models.position_encoding import pos_encode_zoom


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

class Transformer(nn.Module):

    def __init__(self, backbone_zoom, d_model=512, nhead=8, num_encoder_layers=6, num_encoder_zoom_layers = 4,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.backbone_zoom = backbone_zoom

        # encoder image
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        # encoder zoom
        encoder_layer2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm2 = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_zoom = TransformerEncoder(encoder_layer2, num_encoder_zoom_layers, encoder_norm2)


        # decoder 1
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        # decoder 2
        decoder_layer2 = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder2 = TransformerDecoder(decoder_layer2, num_decoder_layers, decoder_norm2,
                                          return_intermediate=return_intermediate_dec)

        self.zoom_coords_embed = MLP(d_model, d_model, 4, 3)
        # Output Embeddings to final Predictions
        self.objectness_embed = nn.Linear(d_model, 1) # only one output class -> objectness
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def get_zoom_features(self, img, zoom_coords, query_mask, size=[50, 50]):   # img:[N, 3, H, W]
        # size: size in px (height, width) of 
        # num_z = torch.max(((visible>0.5) & query_mask).sum(dim=1))
        num_z = query_mask.size(1)
        mask = torch.zeros((img.size(0), num_z), dtype=torch.bool, device=query_mask.device) # [N, num_z]
        fmaps = torch.zeros((img.size(0), num_z, 3, size[0], size[1]), device=img.device)    # [N, seq_len, 3, h_resize, w_resize]
        filtered_coords = torch.zeros((img.size(0), num_z, zoom_coords.size(-1)), device=fmaps.device) # [N, num_z, 4]

        y1 = zoom_coords[:, :, 1] - zoom_coords[:, :, 3]/2
        y1 = y1.clamp(min=0)
        y2 = zoom_coords[:, :, 1] + zoom_coords[:, :, 3]/2
        y2 = y2.clamp(max=1)
        x1 = zoom_coords[:, :, 0] - zoom_coords[:, :, 2]/2
        x1 = x1.clamp(min=0)
        x2 = zoom_coords[:, :, 0] + zoom_coords[:, :, 2]/2
        x2 = x2.clamp(max=1)
        
        tensor_list = []

        for b in range(0, img.size(0)): # batch size
            i = 0
            for s in range(0, zoom_coords.size(1)): # num_q
                # if query_mask[b, s] and visible[b, s] > 0.5:
                    image = img[b].unsqueeze(0)
                    # normalize coords
                    norm_x1 = 2 * x1[b,s] -1
                    norm_x2 = 2 * x2[b,s] -1
                    norm_y1 = 2 * y1[b,s] -1
                    norm_y2 = 2 * y2[b,s] -1
                    # create grid
                    grid_x = torch.linspace(0, 1, size[1], device=img.device) * (norm_x2 - norm_x1) + norm_x1
                    grid_y = torch.linspace(0, 1, size[0], device=img.device) * (norm_y2 - norm_y1) + norm_y1
                    grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
                    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # (1, output_h, output_w, 2)
                    # compute roi
                    roi = F.grid_sample(image, grid, mode='bilinear', align_corners=True)
                    fmaps[b,i,:,:,:] = roi.squeeze(0)
                    tensor_list.append(roi.squeeze(0))
                    mask[b,i] = True
                    filtered_coords[b, i, :] = torch.tensor([y1[b,s], y2[b,s], x1[b,s], x2[b,s]], device=fmaps.device)
                    i += 1
        # result = torch.stack(tensor_list, dim=0).view(query_mask.size(0), query_mask.size(1), 3, size[0], size[1])
        return fmaps, mask, filtered_coords

    def plot_zoom_features(self, zoom_features):
        num_images = zoom_features.size(1)
        for i, image in enumerate(zoom_features[0]):
            img_np = image.permute(1,2,0).cpu().numpy()
            plt.subplot(1, num_images, i+1)
            plt.imshow(img_np)
            plt.title(str(i))

        plt.savefig("zoom_plots.pdf")
        plt.close()
        print("Plot saved")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, src, query_embed, pos_embed, query_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.permute(1, 0, 2)  # from NxSxH to SxNxH

        # run first part of transformer
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(query_embed, memory, pos=pos_embed, tgt_key_padding_mask=query_mask)


        # extract zoom coords based on transformer output
        # zoom_coords_preds = self.zoom_coords_embed(hs[-1].permute(1,0,2)).sigmoid()  # [N, Seq_len (num_q), 4], 4 = [x,y,w,h]
        #
        # zoom_features, zoom_mask, filtered_coords = self.get_zoom_features(images,
                                                                           # zoom_coords_preds,
                                                                           # query_mask)   # [N, seq_len (num_z), 3, h_resize, w_resize], [N, seq_len (num_z)]
        # print()
        # print(visible)
        # print(zoom_coords)
        # self.plot_zoom_features(zoom_features)
        
        # pass zoom features through backbone
        # zoom_features = self.backbone_zoom(zoom_features.flatten(start_dim=0, end_dim=1))   #[NxSeq_len, 3, h_resize, w_resize] -> [NxSeq_len, hidden, h_resize/4, w_resize/4]
        # zoom_pos_encode = pos_encode_zoom(fmap_shape=(zoom_mask.size(0), zoom_mask.size(1), *zoom_features.size()[1:]), 
        #                                   zoom_coords=filtered_coords)  # [N, Seq_len, hidden, h_resize/4, w_resize/4]
        #
        # zoom_embed = zoom_features.flatten(2).permute(2, 0, 1)   # [h_z*w_z, n*seq_len, hidden]
        # n, seq_len, hidden, h_z, w_z = zoom_pos_encode.shape 
        # zoom_pos_embed1 = zoom_pos_encode.flatten(0,1).flatten(2).permute(2, 0, 1) # [h_z*w_z, n*seq_len, hidden]
        # zoom_mask = zoom_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_z, w_z).flatten(1) #[n, seq_len, h_z, w_z] -> [n, seq_len*h_w*w_z]
        

        # run second part of transformer with zoom features
        # memory2 = self.encoder_zoom(zoom_embed, pos=zoom_pos_embed1)  # encoder for zoom features
        # memory2 = memory2.view(h_z*w_z, n, seq_len, hidden).permute(2, 0, 1, 3).flatten(0, 1) # [seq_len*h_z*w_z, n, hidden]
        # memory = torch.cat((memory, memory2)) # [h*w + h_z*w_z*seq_len, n, hidden]
        # zoom_pos_embed2 = zoom_pos_encode.flatten(3).permute(1, 3, 0, 2).flatten(0, 1)    # [seq_len*h_z*w_z, n, hidden]
        # pos_embed = torch.cat((pos_embed, zoom_pos_embed2))
        # memory_mask = torch.ones(size=(memory.size(1), memory.size(0)), dtype=torch.bool, device=memory.device)
        # memory_mask[:, h*w:] = zoom_mask # [n, h*w+seq_len*h*w]
        # memory_mask = ~memory_mask
        # hs2 = self.decoder2(hs[-1], memory, pos=pos_embed, memory_key_padding_mask=memory_mask, tgt_key_padding_mask=query_mask)
        hs2 = self.decoder2(hs[-1], memory, pos=pos_embed, tgt_key_padding_mask=query_mask)
        out = torch.cat((hs, hs2)).transpose(1,2)
        outputs_objectness = self.objectness_embed(out).sigmoid().squeeze(dim=-1) # [Num_Decoding, N, Seq_len]
        outputs_coord = self.bbox_embed(out).sigmoid() # [Num_Decoding, N, Seq_len, 4]

        return outputs_objectness, outputs_coord


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = tgt
        tgt_mask = torch.tensor([tgt_mask])
        tgt_key_padding_mask = ~tgt_key_padding_mask # flip target mask
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = tgt2
        tgt_key_padding_mask = ~tgt_key_padding_mask # flip target mask
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
