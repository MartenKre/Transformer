from models.detr import DETR, SetCriterion
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 


def init_position_encoding(hidden_dim):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding


def init_backbone(lr_backbone, hidden_dim, backbone='resnet50', dilation=False):
    # masks are only used for image segmentation

    position_embedding = init_position_encoding(hidden_dim)
    train_backbone = lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )
    
def init_V2W_Transformer(lr_backbone=1e-5, hidden_dim=256, dropout=.1, nheads=8, dim_feedforward=2048, enc_layers=6,
                         dec_layers=6, pre_norm=True, aux_loss=False, use_embeddings=False):
    backbone = init_backbone(lr_backbone, hidden_dim)
    transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
    model = DETR(
        backbone,
        transformer,
        input_dim_gt=2,
        aux_loss=aux_loss,
        use_embeddings=use_embeddings,
    )
    return model

