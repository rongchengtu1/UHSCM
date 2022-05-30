import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import CalcHammingRanking as CalcHR
import torch.nn.functional as F
import scipy.io as scio
import h5py

from datetime import datetime
from hash_model import HASH_Net
from torch.utils.data import DataLoader
import dataset_my as dp
import cv2

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def GenerateCode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, _, data_ind = data
        data_input = data_input.cuda()
        output = model(data_input)
        B[data_ind.numpy(), :] = output.cpu().data.numpy()

    return B

def UHSCM_algo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    code_length = opt.bit
    bit = opt.bit

    lamda = opt.lamda
    temp = opt.temp
    sc_f = opt.scf
    alpha = opt.alpha
    ablation_type = opt.ablation_type
    batch_size = 128
    epochs = 60
    best_map = 0.

    learning_rate_i = 0.006

    weight_decay = 10 ** -5
    model_name = 'vgg19'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.7),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(3),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ### data processing
    if 'nus' in opt.data_set:
        all_dta = h5py.File(opt.data_path + '/nus.h5')
        dset_database = dp.DatasetPorcessing_nus_h5(
            np.asarray(all_dta['data_set']), np.asarray(all_dta['dataset_L']), transformations)
        dset_train = dp.DatasetPorcessing_nus_h5(
            np.asarray(all_dta['train_data']), np.asarray(all_dta['train_L']), train_transforms, True)
        dset_test = dp.DatasetPorcessing_nus_h5(
            np.asarray(all_dta['test_data']), np.asarray(all_dta['test_L']), transformations)
    elif 'cifar' in opt.data_set:
        all_dta = scio.loadmat(opt.data_path + '/cifar-10.mat')
        dset_database = dp.DatasetPorcessing_mat(
            all_dta['data_set'], all_dta['dataset_L'], transformations)
        dset_train = dp.DatasetPorcessing_mat(
            all_dta['train_data'], all_dta['train_L'], train_transforms, True)
        dset_test = dp.DatasetPorcessing_mat(
            all_dta['test_data'], all_dta['test_L'], transformations)
    elif 'flickr' in opt.data_set:
        all_dta = scio.loadmat(opt.data_path + '/FLICKR25K.mat')
        dset_database = dp.DatasetPorcessing_flickr_mat(
            all_dta['data_set'], all_dta['dataset_L'], transformations)
        dset_train = dp.DatasetPorcessing_flickr_mat(
            all_dta['train_data'], all_dta['train_L'], train_transforms, True)
        dset_test = dp.DatasetPorcessing_flickr_mat(
            all_dta['test_data'], all_dta['test_L'], transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4
                                 )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )



    ### create model
    model = HASH_Net(model_name, code_length)
    model.cuda()
    optimizer = torch.optim.SGD([{'params': model.features.parameters(), 'lr': learning_rate_i * 0.05},
                                       {'params': model.classifier.parameters(), 'lr': learning_rate_i * 0.05},
                                       {'params': model.hash_layer.parameters(), 'lr': learning_rate_i}],
                                      lr=learning_rate_i, weight_decay=weight_decay)

    ### training phase
    # parameters setting

    train_labels_onehot = dset_train.labels
    test_labels_onehot = dset_test.labels
    database_labels_onehot = dset_database.labels

    with open(opt.sim_path) as f:
        final_sim = pickle.load(f)
        probs = (image_label_probs > 0.5).float()
        image_label_probs = F.normalize(torch.tensor(image_label_probs).float())
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_lossc = 0.0
        epoch_lossr = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_input, train_input1, _, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)

            train_label_onehot = train_label.type(torch.FloatTensor)
            train_input, train_input1, train_label = train_input.cuda(), train_input1.cuda(), train_label.cuda()
            S = final_sim[batch_ind, :]
            S = S[:, batch_ind]

            model.zero_grad()
            train_outputs1 = model(train_input)
            # train_outputs2 = model(train_input1)

            Bbatch = torch.sign(train_outputs1)
            regterm = (Bbatch - train_outputs1).pow(2).sum() / len(train_label)
            train_outputs = F.normalize(train_outputs1)
            SC = (S >= sc_f).float().cuda()

            theta_exp = torch.exp(train_outputs.mm(train_outputs.t()) / temp)
            the_frac = ((1 - SC) * theta_exp).sum(1).view(-1, 1) + 0.00001
            loss_c = - (torch.log(theta_exp / the_frac) * SC).sum() / SC.sum()
            # print(loss_c.data.cpu())

            theta_x = train_outputs.mm(train_outputs.t())
            logloss = (theta_x - S.cuda()).pow(2).sum() / (len(train_label) * len(train_label))


            loss = logloss + lamda * regterm + alpha * loss_c
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_lossc += loss_c.item()
            epoch_lossr += regterm.item()

        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f][Lossc: %3.5f][Lossr: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader), epoch_lossr / len(train_loader), epoch_lossc / len(train_loader)))
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate_i)


        ### testing during epoch
        if (epoch + 1) % 20 == 0:
            model.eval()
            qB = GenerateCode(model, test_loader, num_test, bit)
            dB = GenerateCode(model, database_loader, num_database, bit)
            # map_ = CalcHR.CalcMap(np.sign(qB), np.sign(dB), test_labels_onehot.numpy(), database_labels_onehot.numpy())
            map_5000 = CalcHR.CalcTopMap(np.sign(qB), np.sign(dB), test_labels_onehot.numpy(), database_labels_onehot.numpy(), 5000)
            if best_map < map_5000:
                best_map = map_5000
            model.train()

            print('[Test Phase ][Epoch: %3d/%3d] MAP_5000(retrieval train): %3.5f BEST_MAP(retrieval train): %3.5f' % (epoch + 1, epochs, map_5000, best_map))
    ### evaluation phase
    model.eval()
    qB = GenerateCode(model, test_loader, num_test, bit)
    dB = GenerateCode(model, database_loader, num_database, bit)

    map_5000 = CalcHR.CalcTopMap(np.sign(qB), np.sign(dB), test_labels_onehot.numpy(), database_labels_onehot.numpy(), 5000)
    print('MAP_5000(retrieval): %3.5f BEST_MAP(retrieval): %3.5f' % (map_5000, best_map))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--scf", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lamda", default=0.001, type=float)
    parser.add_argument("--data_set", default=0, type=str)
    parser.add_argument("--sim_path", default=0, type=str)
    parser.add_argument("--data_path", default=0, type=str)
    opt = parser.parse_args()
    UHSCM_algo(opt)




