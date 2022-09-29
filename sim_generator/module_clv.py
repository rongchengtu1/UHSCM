import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pickle
import numpy as np
from clip import clip
from clip.model import convert_weights
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from module_utils import Block, CrossEn
_tokenizer = _Tokenizer()


def load_clip_to_cpu(model_path='/apdcephfs/share_1367250/rongchengtu/CLIP4Clip_public/modules/ViT-B-16.pt'):
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict)

    return model
class CLIP_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_source = load_clip_to_cpu()
        convert_weights(self.clip_source)

    def forward(self, image):
        image_features = self.clip_source.encode_image(image.half())
        return image_features
