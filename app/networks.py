import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
# from models import create_model
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import numpy as np
import random
import os
import sys
from itertools import repeat

class FeatureExtractor(nn.Module):
    """
    This class defines the feature extractor model which in our case is 
    a foundation model named H-optimus-0.
    I do not fine-tune the whole model, I freeze the model layers, 
    add a linear layer to the model then I finetune parameters of added layer.
    """
    def __init__(self, args, device, custom_weights_path=None):
        super(FeatureExtractor, self).__init__()
        self.feature_dim=1536
        
        if args.model == 'H-optimus-0':
            self.backbone = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
            )
            class my_model(nn.Module):
                def __init__(self, model, feature_dim):
                    super(my_model, self).__init__()
                    self.model = model
                    if args.add_layer == 'True':
                        self.nl = nn.Linear(1536, feature_dim)
                def forward(self, img):
                    features = self.model(img)
                    if args.add_layer == 'True':
                        features = self.nl(features)
                        return features
                    else:
                        return features
            self.backbone = my_model(self.backbone, self.feature_dim)
        else:
            print('Error: unsupported model')
            exit()

        self.backbone.head = nn.Identity()

        if args.add_layer == 'True':
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.nl.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
