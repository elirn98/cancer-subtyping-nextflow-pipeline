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
from huggingface_hub import login, hf_hub_download

class FeatureExtractor(nn.Module):
    def __init__(self, args, device, custom_weights_path=None):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained models
        if args.model == "convnext_base" or args.model == "vit" or args.model == "CTransPath" or args.model == "Phikon":
            self.feature_dim=768
        elif args.model == "PLIP":
            self.feature_dim=512
        elif args.model == "resnet50":
            self.feature_dim=2048
        elif args.model == "swin":
            self.feature_dim=1024
        elif args.model == "Lunit-Dino":
            self.feature_dim=384
        elif args.model == "H-optimus-0":
            self.feature_dim=1536
        elif args.model == "Virchow2" or args.model == "UNI":
            self.feature_dim=1024

        if args.model == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True).to(device)
            if custom_weights_path != None:
                self.backbone.fc = nn.Linear(2048, args.num_classes)
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                if 'net' in weights.keys():
                    weights = weights['net']
                elif 'state_dict' in weights.keys():
                    weights = weights['state_dict']
                    try:
                        self.backbone.load_state_dict(weights)
                    except:
                        self.backbone = load_state_dict_with_prefix(self.backbone, weights, prefix='module.model.')
            self.backbone.fc = nn.Sequential()
            #print(model)
        # elif args.model == 'resnet50_cds':
        #     sys.path.append('/projects/ovcare/classification/Ali/Search_Engine/cross_domain_adaptation/codes/CDS/CDS_pretraining')
        #     inc = 2048
        #     self.backbone = models.__dict__['resnet50'](pretrained=True, low_dim=512)
        #     #model = nn.DataParallel(model)
        #     if custom_weights_path != None:
        #         weights = torch.load(custom_weights_path, map_location=torch.device(device))
        #         weights = weights['net']
        #         self.backbone.load_state_dict(weights, strict=False)
        #     self.backbone.fc = nn.Sequential()
            #print(model)
        elif args.model == 'convnext_base':
            self.backbone = torchvision.models.convnext_small(pretrained=True)
            if custom_weights_path != None:
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                #weights = weights['state_dict']
                self.backbone.load_state_dict(weights, strict=False)
            self.backbone.classifier[2] = nn.Sequential()
        elif args.model == 'densenet121':
            self.backbone = torchvision.models.densenet121(pretrained=True).to(device)
            self.backbone.features = nn.Sequential(self.backbone.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
            self.backbone.classifier = nn.Sequential()
        elif args.model == 'KimiaNet':
            class my_model(nn.Module):
                def __init__(self, model):
                    super(my_model, self).__init__()
                    self.model = model
                def forward(self, img):
                    return self.model(img)[0]
            class fully_connected(nn.Module):
                def __init__(self, model, num_ftrs, num_classes):
                    super(fully_connected, self).__init__()
                    self.model = model
                    self.fc_4 = nn.Linear(num_ftrs,num_classes)
                
                def forward(self, x):
                    x = self.model(x)
                    x = torch.flatten(x, 1)
                    out_1 = x
                    out_3 = self.fc_4(x)
                    return  out_1, out_3
            model = torchvision.models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
            num_ftrs = model.classifier.in_features
            model = fully_connected(model.features, num_ftrs, 30)
            model = model.to(device)
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load('/projects/ovcare/classification/Ali/Ovarian_project/Pytorch_Codes/KimiaNet/KimiaNetPyTorchWeights.pth', map_location=torch.device('cpu')))
            model.module.fc_4 = nn.Sequential()
            model = my_model(model)
            if custom_weights_path != None:
                model.head = nn.Linear(1024, num_classes)
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                if 'net' in weights.keys():
                    weights = weights['net']
                elif 'state_dict' in weights.keys():
                    weights = weights['state_dict']
                    try:
                        model.load_state_dict(weights)
                    except:
                        model = load_state_dict_with_prefix(model, weights, prefix='module.model.module.model.', per='model.module.model.', strict=False)
            model.head = nn.Identity()
            #print(model)
        elif args.model == 'vit' or args.model == 'vit_b16':
            print('Using ViT')
            self.backbone = torchvision.models.vit_b_16(pretrained=True).to(device)
            if custom_weights_path != None:
                self.backbone.head = nn.Linear(768, args.num_classes)
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                if 'net' in weights.keys():
                    weights = weights['net']
                elif 'state_dict' in weights.keys():
                    weights = weights['state_dict']
                    try:
                        self.backbone.load_state_dict(weights)
                    except:
                        self.backbone = load_state_dict_with_prefix(self.backbone, weights, prefix='module.')
            self.backbone.heads = nn.Identity()
            #print(model)
        elif args.model == 'swin' or args.model == 'swin_b':
            self.backbone = torchvision.models.swin_b(pretrained=True).to(device)
            if custom_weights_path != None:
                self.backbone.head = nn.Linear(1024, args.num_classes)
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                if 'net' in weights.keys():
                    weights = weights['net']
                elif 'state_dict' in weights.keys():
                    weights = weights['state_dict']
                    try:
                        self.backbone.load_state_dict(weights)
                    except:
                        self.backbone = load_state_dict_with_prefix(self.backbone, weights, prefix='module.')
            self.backbone.head = nn.Sequential()
            #print(model)
        elif args.model == 'swin_t':
            self.backbone = torchvision.models.swin_t(pretrained=True).to(device)
            if custom_weights_path != None:
                self.backbone.head = nn.Linear(768, args.num_classes)
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                if 'net' in weights.keys():
                    weights = weights['net']
                elif 'state_dict' in weights.keys():
                    weights = weights['state_dict']
                    try:
                        self.backbone.load_state_dict(weights)
                    except:
                        self.backbone = load_state_dict_with_prefix(self.backbone, weights, prefix='module.')
            self.backbone.head = nn.Sequential()
        elif args.model == 'PLIP':
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            class my_model(nn.Module):
                def __init__(self, model):
                    super(my_model, self).__init__()
                    self.model = model
                def forward(self, img):
                    return self.model.get_image_features(pixel_values = img)

            def plip(pretrained=True, low_dim=512):
                """Constructs a ViT-b16 model.

                Args:
                    pretrained (bool): If True, returns a model pre-trained on ImageNet
                """
                from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
                processor = AutoProcessor.from_pretrained("vinid/plip")
                model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
                model = my_model(model)
                return model
            processor = AutoProcessor.from_pretrained("vinid/plip")
            self.backbone = plip()
            if custom_weights_path != None:
                weights = torch.load(custom_weights_path, map_location=torch.device(device))
                weights = weights['net']
                self.backbone.load_state_dict(weights)
        # elif args.model == 'CTransPath':
        #     sys.path.append('/projects/ovcare/classification/Feature_Extractors_Weights/ctranspath/timm-0.5.4/timm')
        #     #from models.layers.helpers import to_2tuple

        #     # From PyTorch internals
        #     def _ntuple(n):
        #         def parse(x):
        #             if isinstance(x, collections.abc.Iterable):
        #                 return x
        #             return tuple(repeat(x, n))
        #         return parse

        #     to_2tuple = _ntuple(2)
        #     def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
        #         min_value = min_value or divisor
        #         new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        #         # Make sure that round down does not go down by more than 10%.
        #         if new_v < round_limit * v:
        #             new_v += divisor
        #         return new_v

        #     class ConvStem(nn.Module):
        #         def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        #             super().__init__()

        #             assert patch_size == 4
        #             assert embed_dim % 8 == 0

        #             img_size = to_2tuple(img_size)
        #             patch_size = to_2tuple(patch_size)
        #             self.img_size = img_size
        #             self.patch_size = patch_size
        #             self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        #             self.num_patches = self.grid_size[0] * self.grid_size[1]
        #             self.flatten = flatten


        #             stem = []
        #             input_dim, output_dim = 3, embed_dim // 8
        #             for l in range(2):
        #                 stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
        #                 stem.append(nn.BatchNorm2d(output_dim))
        #                 stem.append(nn.ReLU(inplace=True))
        #                 input_dim = output_dim
        #                 output_dim *= 2
        #             stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        #             self.proj = nn.Sequential(*stem)

        #             self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        #         def forward(self, x):
        #             B, C, H, W = x.shape
        #             assert H == self.img_size[0] and W == self.img_size[1], \
        #                 f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #             x = self.proj(x)
        #             if self.flatten:
        #                 x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #             x = self.norm(x)
        #             return x
            
        #     self.backbone = create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        #     self.backbone.head = nn.Identity()
        #     td = torch.load('/projects/ovcare/classification/Feature_Extractors_Weights/ctranspath/ctranspath.pth')
        #     self.backbone.load_state_dict(td['model'], strict=False)
        elif args.model == 'Phikon':
            from transformers import AutoImageProcessor, AutoModel
            class my_model(nn.Module):
                def __init__(self, model, feature_dim):
                    super(my_model, self).__init__()
                    self.model = model
                    if args.add_layer == 'True':
                        self.nl = nn.Linear(768, feature_dim)
                def forward(self, img):
                    features = self.model(img)[1]
                    if args.add_layer == 'True':
                        features = self.nl(features)
                        return features
                    else:
                        return features
            processor = AutoImageProcessor.from_pretrained("owkin/phikon")
            self.backbone = AutoModel.from_pretrained("owkin/phikon")
            self.backbone = my_model(self.backbone, self.feature_dim)
        elif args.model == 'Lunit-Dino':
            def get_pretrained_url(key):
                URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
                model_zoo_registry = {
                    "DINO_p16": "dino_vit_small_patch16_ep200.torch",
                    "DINO_p8": "dino_vit_small_patch8_ep200.torch",
                }
                pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
                return pretrained_url

            def vit_small(pretrained, progress, key, **kwargs):
                patch_size = kwargs.get("patch_size", 16)
                model = VisionTransformer(
                    img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
                )
                if pretrained:
                    pretrained_url = get_pretrained_url(key)
                    verbose = model.load_state_dict(
                        torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
                    )
                    print(verbose)
                return model
            self.backbone = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        elif args.model == 'UNI':
            import sys
            login(token="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")
            # self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True, use_auth_token=True)
            # self.backbone.eval()
            local_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
            os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
            hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
            self.backbone = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.backbone.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
            self.backbone.eval()
            if args.add_layer == 'True':
                for param in self.backbone.parameters():
                    param.requires_grad = False
            class my_model(nn.Module):
                def __init__(self, model, feature_dim):
                    super(my_model, self).__init__()
                    self.model = model
                    if args.add_layer == 'True':
                        self.nl = nn.Linear(1024, feature_dim)
                def forward(self, img):
                    with torch.no_grad():
                        features = self.model(img)
                    if args.add_layer == 'True':
                        features = self.nl(features)
                        return features
                    else:
                        return features
            self.backbone = my_model(self.backbone, self.feature_dim)

        elif args.model == 'H-optimus-0':
            # login("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")
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
        elif args.model == 'Virchow2':
            self.backbone = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).to(device)
            self.backbone.eval()  # Keep the backbone in evaluation mode to prevent batchnorm updates
            # Freeze all parameters of the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            class my_model(nn.Module):
                def __init__(self, model, feature_dim):
                    super(my_model, self).__init__()
                    self.model = model
                    if args.add_layer == 'True':
                        self.nl = nn.Linear(2560, feature_dim)
                def forward(self, img):
                    with torch.no_grad():
                        output = self.model(img)
                    class_token = output[:, 0]    # size: 1 x 1280
                    patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

                    # concatenate class token and average pool of patch tokens
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

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
            # for param in self.backbone.model.parameters():
            #     param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Only the new layer's parameters will be updated
            for param in self.backbone.nl.parameters():
                param.requires_grad = True
        # for name, param in self.backbone.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        #         print(param)

    def forward(self, x):
        return self.backbone(x)


class FeatureHead(nn.Module):
    def __init__(self, args):
        super(FeatureHead, self).__init__()
        if args.model == "convnext_base" or args.model == "vit" or args.model == "CTransPath" or args.model == "Phikon":
            self.feature_dim=768
        elif args.model == "PLIP":
            self.feature_dim=512
        elif args.model == "resnet50":
            self.feature_dim=2048
        elif args.model == "swin":
            self.feature_dim=1024
        elif args.model == "Lunit-Dino":
            self.feature_dim=384
        elif args.model == 'H-optimus-0':
            self.feature_dim=1536
        elif args.model == 'Virchow2':
            self.feature_dim=10
        input_dim = self.feature_dim
        hidden_dim = (2 * input_dim) / 3
        output_dim = hidden_dim
        self.feature_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
