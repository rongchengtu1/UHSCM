import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.io as scio
import h5py
from torch.cuda.amp import autocast as autocast, GradScaler
from clip.tokenization_clip import SimpleTokenizer as ClipTokenizer
from datetime import datetime
from module_clv import CLIP_f as feature_extractor
from torch.utils.data import DataLoader
import dataset_my as dp
import cv2


def Generatefeature(model, data_loader, num_data, bit):
    B = torch.zeros((num_data, bit))
    for iter, data in enumerate(data_loader, 0):
        data_input, label, _, data_ind = data
        data_input = data_input.cuda()
        with autocast():
            output = model(data_input)
        B[data_ind.numpy(), :] = output.cpu().data.float()
    return B

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ### data processing
    if 'nus' in opt.data_set:
        all_dta = h5py.File(opt.data_path, 'r')
        dset_train = dp.DatasetPorcessing_nus_h5(
            np.asarray(all_dta['train_data']), np.asarray(all_dta['train_L']), transformations)
    elif 'cifar' in opt.data_set:
        all_dta = scio.loadmat(opt.data_path)
        dset_train = dp.DatasetPorcessing_mat(
            all_dta['train_data'], all_dta['train_L'], transformations)
    elif 'flickr' in opt.data_set:
        all_dta = scio.loadmat(opt.data_path)
        dset_train = dp.DatasetPorcessing_flickr_mat(
            all_dta['train_data'], all_dta['train_L'], transformations)

    num_train = len(dset_train)

    train_loader = DataLoader(dset_train,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=4
                              )


    ### create model
    model = feature_extractor()
    model.cuda()

    ### extract feature
    image_feature = Generatefeature(model, train_loader, num_train, 512)

    tokenizer = ClipTokenizer()
    max_words = 77
    f = open(opt.concept_path, 'r')
    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                          "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    concept_prompt = ['a photo of the ' + x.strip() for x in f.readlines()]
    text_inputs = np.zeros((len(concept_prompt), max_words), dtype=np.long)
    f.close()
    for i in range(len(concept_prompt)):
        text = concept_prompt[i]
        words = tokenizer.tokenize(text)
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = tokenizer.convert_tokens_to_ids(words)
        while len(input_ids) < max_words:
            input_ids.append(0)

        text_inputs[i] = np.array(input_ids)

    text_inputs = torch.tensor(text_inputs).cuda()
    with torch.no_grad():
        text_features = model.clip_source.encode_text(text_inputs).cpu()

    ### concept denoising
    temp = text_features.shape[0] * opt.tua
    image_feature = F.normalize(image_feature).cuda()
    text_features = F.normalize(text_features.float()).cuda()

    logits_per_image = image_feature.mm(text_features.t()) * temp

    probs = logits_per_image.softmax(dim=-1).cpu()

    max_concept_index = probs.max(dim=1)[1]
    concept_num = [0] * probs.shape[1]
    for i in range(probs.shape[0]):
        concept_num[max_concept_index[i]] += 1

    concept_num = torch.Tensor(concept_num)
    select_concept = (concept_num > ((probs.shape[0] / float(probs.shape[1])) * 0.5)).float()

    ### gnerate similarity matrix
    temp = select_concept.sum() * opt.tua
    logits_per_image = image_feature.mm(text_features.t()) * temp
    logits_per_image = logits_per_image.masked_fill((1 - select_concept).bool().cuda(), -float('inf'))
    probs = logits_per_image.softmax(dim=-1).cpu()

    norm_probs = F.normalize(probs)
    sim = 2. * norm_probs.mm(norm_probs.t()) - 1.

    with open(opt.sim_path, 'wb') as f:
        pickle.dump(sim, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=6195)  # 6195
    parser.add_argument('--tua', type=float, default=3)  # 6195
    parser.add_argument('--batch_size', type=int, default=10)  # 6195
    parser.add_argument("--data_set", type=str)
    parser.add_argument("--concept_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--sim_path", type=str)
    opt = parser.parse_args()
    main(opt)




