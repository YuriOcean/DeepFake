import torch.nn as nn
import torch
import torch.nn.functional as F

from ForensicHub.registry import MODELS, build_from_registry, register_model
from ForensicHub.core.base_model import BaseModel

'''
change data input: [image, mask, label] -> [image, label]
add max pooling detection head and use label for loss only
'''


@register_model("Mask2LabelWrapper")
class Mask2LabelWrapper(BaseModel):
    def __init__(self, name='Resnet50', init_config={}):
        super().__init__()
        self.name = name
        dict = {"name": name, "init_config": init_config}
        self.base_model = build_from_registry(MODELS, dict)
        self.head = nn.AdaptiveMaxPool2d(1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, image, label, *args, **kwargs):
        if not 'mask' in kwargs:
            mask = torch.randint(0, 1, (image.shape[0], 1, image.shape[2], image.shape[3])).long().to(image.device)
        else:
            mask = kwargs['mask']
        if not 'edge_mask' in kwargs:
            edge_mask = torch.randint(0, 1, (image.shape[0], 1, image.shape[2], image.shape[3])).long().to(image.device)
        else:
            edge_mask = kwargs['edge_mask']
        label = label.float()
        if self.name in ['IML_ViT', 'Mesorch', 'Cat_Net']:
            if self.name == 'Cat_Net':
                features = self.base_model.forward_features(image=image, mask=mask, edge_mask=edge_mask, label=label,
                                                            DCT_coef=kwargs['DCT_coef'], qtables=kwargs['qtables'])
            else:
                features = self.base_model.forward_features(image=image, mask=mask, edge_mask=edge_mask, label=label)
            if type(features) == tuple:
                features = features[0]
            pred_label = self.head(features)
            if pred_label.shape[1] != 1:
                pred_label = pred_label[:, 1].view(-1)
            else:
                pred_label = pred_label.view(-1)
            loss = self.loss_fn(pred_label, label.float())
            pred_label = F.sigmoid(pred_label)
        elif self.name in ['DTD', 'FFDN']:
            out_dict = self.base_model(image=image, mask=mask, dct=kwargs['dct'], qt=kwargs['qt'])
            pred_label = self.head(out_dict['pred_mask'][:, 0:1, :, :])
            pred_label = pred_label.view(-1)
            loss = F.binary_cross_entropy(pred_label, label)
        else:
            outputs = self.base_model(image=image, mask=mask, edge_mask=edge_mask, label=label)
            pred_label = outputs['pred_label']
            loss = F.binary_cross_entropy(pred_label, label.float())
        # ----------Output interface--------------------------------------
        output_dict = {
            "backward_loss": loss,
            "pred_label": pred_label,
            "visual_loss": {
                'pred_loss': loss
            }
        }
        return output_dict
