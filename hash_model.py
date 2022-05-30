import torch.nn as nn
from torchvision import models
import torch

LAYER1_NODE = 40960


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0.)

class HASH_Net(nn.Module):
    def __init__(self, model_name, bit, pretrained=True):
        super(HASH_Net, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias
            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True)
            )
            self.hash_layer = nn.Sequential(
                cl3,
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg19":
            original_model = models.vgg19(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)

            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.hash_layer = nn.Sequential(
                cl3,
                nn.Tanh()
            )
            self.model_name = 'vgg19'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
            f = self.classifier(f)
            f = self.hash_layer(f)
        else:
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
            f = self.hash_layer(f)
        return f

