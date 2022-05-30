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
    def __init__(self, bit, path='/apdcephfs/share_1367250/rongchengtu/coco/'):
        super().__init__()
        self.clip_source = load_clip_to_cpu()
        convert_weights(self.clip_source)

    def forward(self, image):
        image_features = self.clip_source.encode_image(image.half())
        return image_features

class CustomCLIP(nn.Module):
    def __init__(self, bit, path='/apdcephfs/share_1367250/rongchengtu/coco/'):
        super().__init__()
        self.clip_source = load_clip_to_cpu()
        self.clip_ori = copy.deepcopy(self.clip_source)
        convert_weights(self.clip_source)
        convert_weights(self.clip_ori)
        self.dtype = self.clip_source.dtype
        # self.logit_scale = (torch.ones([])*np.log(1/0.01)).cuda()
        self.hash_layer = nn.Linear(512, bit)
        self.patch_weight = nn.Linear(512, 1)
        self.hash_layer.apply(self._init_weights)
        self.patch_weight.apply(self._init_weights)
        category_feature = pickle.load(open(path + 'category_text_clip_base32_feature.pkl', 'rb'))
        category_feature = (category_feature / category_feature.norm(dim=-1, keepdim=True))
        self.category_feature = category_feature.t().view(1, category_feature.shape[1], -1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)
        # self.avg = nn.AvgPool2d(kernel_size=3, stride=2)
        # self.avg = nn.AvgPool2d(kernel_size=4, stride=3)

        # self.text_projection = nn.Parameter(torch.empty(512, arg.feature_dim))
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.clip_target.transformer.width ** -0.5)
        # convert_weights(self.text_projection)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def source_img_text(self, image, text, text_mask):
        image_features = self.clip_source.encode_image(image.type(self.dtype))

        text_features = self.clip_source.encode_text(text).float()

        ### mean pooling
        text_features = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)]
        pool_visual = (image_features / image_features.norm(dim=-1, keepdim=True)).mean(dim=1)
        norm_visual = pool_visual / pool_visual.norm(dim=-1, keepdim=True)
        mean_feafure = (text_features + norm_visual) / 2.
        norm_mean_feature = mean_feafure / mean_feafure.norm(dim=-1, keepdim=True)
        return norm_mean_feature

        # ### cross interaction
        # image_features = self.ln_final1(image_features.type(torch.half))
        # text_features = self.ln_final2(text_features.type(torch.half))
        # cls_token = self.cls_token + self.cls_token_mask
        # cls_tokens = cls_token.expand(image_features.shape[0], -1, -1)
        # # print(text_features.shape, image_features.shape, self.visual_token_mask.shape, self.cls_token.shape, self.visual_token_mask.shape, self.text_token_mask.shape)
        # img_text = torch.cat(
        #     (cls_tokens, image_features + self.visual_token_mask, text_features + self.text_token_mask), dim=1)
        #
        # visual_mask = torch.ones((image_features.size(0), image_features.size(1) + 1)).to(torch.bool).to(
        #     image_features.device)
        # mask = torch.cat((visual_mask, text_mask), dim=1)
        # for blk in self.blocks:
        #     img_text = blk(img_text, mask)
        #
        # norm_cls_img_text = img_text[:, 0, :] / img_text[:, 0, :].norm(dim=-1, keepdim=True)
        #
        # return norm_cls_img_text

    def target_img_feature(self, image_target):
        image_target_features = self.clip_target.encode_image(image_target.type(self.dtype))
        pool_target_visual = (image_target_features / image_target_features.norm(dim=-1, keepdim=True)).mean(dim=1)
        norm_target_visual = pool_target_visual / pool_target_visual.norm(dim=-1, keepdim=True)
        return norm_target_visual

    def forward(self, image, label, weighted=True):
        image_features1 = self.clip_source.encode_image(image.type(self.dtype))
        image_features_ori = self.clip_ori.encode_image(image.type(self.dtype))
        cls_feature = image_features1[:, 0, :]
        cls_feature = (cls_feature / cls_feature.norm(dim=-1, keepdim=True))

        image_features_ = image_features1[:, 1:, :]
        image_features_ = image_features_.permute(0, 2, 1)
        image_features_ = image_features_.reshape(image_features_.shape[0], image_features_.shape[1], 7, 7)
        image_features_ = self.avg(image_features_)
        image_features_ = image_features_.reshape(image_features_.shape[0], image_features_.shape[1], -1)
        image_features_ = image_features_.permute(0, 2, 1)

        image_features = image_features_ori[:, 1:, :]
        image_features = image_features.permute(0, 2, 1)
        image_features = image_features.reshape(image_features.shape[0], image_features.shape[1], 7, 7)

        image_features = self.avg(image_features)
        image_features = image_features.reshape(image_features.shape[0], image_features.shape[1],-1)
        image_features = image_features.permute(0, 2, 1)


        norm_features = (image_features / image_features.norm(dim=-1, keepdim=True))
        # logit_scale = self.logit_scale.exp()
        logits_per_image = 100 * norm_features.bmm(self.category_feature.repeat(norm_features.shape[0], 1, 1).cuda())
        label_weight = self.softmax(logits_per_image).detach()
        max_weight_index = label_weight.argmax(dim=-1)
        b, n, c = label_weight.shape
        label_weight = label_weight.view(-1, c).contiguous()
        max_weight = label_weight[torch.arange(b * n), max_weight_index.view(-1)]
        label = label.view(b, 1, c)
        label = label.repeat(1, n, 1).view(-1, c).contiguous()
        is_label = label[torch.arange(b * n), max_weight_index.view(-1)]

        weight = self.sigmoid(self.patch_weight(image_features_))


        final_weight = (1 - weight.view(-1)) * (1 - is_label) + is_label * weight.view(-1)
        # loss_weight = - (torch.log(final_weight) * max_weight).sum() / max_weight.sum()

        norm_features_ = (image_features_ / image_features_.norm(dim=-1, keepdim=True))
        # # print(norm_features_.shape, 'nor')
        if weighted:
            pool_visual = (norm_features_ * weight).sum(dim=1) / (weight.sum(dim=1) + 1e-6)
        else:
            pool_visual = norm_features_.mean(1)
        final_feature = cls_feature + pool_visual
        # print(pool_visual.shape, norm_features_.shape, weight.shape, cls_feature.shape, final_feature.shape)
        code = torch.tanh(self.hash_layer(final_feature))

        return code, final_weight, max_weight

        # image_features = self.ln_final1(image_features.type(torch.half))
        # text_features = self.ln_final2(text_features.type(torch.half))
        #
        # cls_token = self.cls_token + self.cls_token_mask
        # cls_tokens = cls_token.expand(image_features.shape[0], -1, -1)
        # # print(text_features.shape, image_features.shape, self.visual_token_mask.shape, self.cls_token.shape, self.visual_token_mask.shape, self.text_token_mask.shape)
        # img_text = torch.cat((cls_tokens, image_features+self.visual_token_mask, text_features+self.text_token_mask), dim=1)
        #
        # visual_mask = torch.ones((image_features.size(0), image_features.size(1)+1)).to(torch.bool).to(image_features.device)
        # mask = torch.cat((visual_mask, text_mask), dim=1)
        # for blk in self.blocks:
        #     img_text = blk(img_text, mask)
        #
        # pool_target_visual = (image_target_features / image_target_features.norm(dim=-1, keepdim=True)).mean(dim=1)
        # norm_cls_img_text = img_text[:, 0, :] / img_text[:, 0, :].norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = pool_target_visual.mm(norm_cls_img_text.t()) * logit_scale
        # loss_vtcontrastive = self.loss_crossen(logits) + self.loss_crossen(logits.T)
        #
        #
        # return loss_vtcontrastive

