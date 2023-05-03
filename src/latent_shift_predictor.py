import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np
from vit_pytorch.deepvit import DeepViT


def save_hook(module, input, output):
    setattr(module, 'output', output)


class LatentShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        if x1.shape[1] == 1 and x2.shape[1] == 1:
            self.features_extractor(torch.cat([x1.repeat([1,3,1,1]), x2.repeat([1,3,1,1])], dim=1))
        else:
            self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class LatentDiffShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentDiffShiftPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(3, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x_diff = x2 - x1
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x_diff = F.interpolate(x_diff, self.downsample)
        if x_diff.shape[1] == 1:
            self.features_extractor(x_diff.repeat([1,3,1,1]))
        else:
            self.features_extractor(x_diff)
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class LeNetDiffShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetDiffShiftPredictor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        x_diff = x2 - x1
        batch_size = x1.shape[0]
        features = self.convnet(x_diff)
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


class LeNetDiffShallowShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetDiffShallowShiftPredictor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(8 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(8 * width, 1),
        )

    def forward(self, x1, x2):
        x_diff = x2 - x1
        batch_size = x1.shape[0]
        features = self.convnet(x_diff)
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


class LeNetShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetShiftPredictor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.convnet(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


class FCShiftPredictor(nn.Module):
    def __init__(self,input_dim, inner_dim, dim):
        super(FCShiftPredictor, self).__init__()

        self.fc_logits = nn.Sequential(
            nn.Linear(2*input_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(2*input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = torch.cat([x1, x2], dim=1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


class ViTShiftPredictor(nn.Module):
    def __init__(self,image_size, dim, channels=6, patch_size=32):
        super(ViTShiftPredictor, self).__init__()
        self.dim = dim
        self.v = DeepViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = self.dim+1,
            dim = 1024,
            depth = 6,
            heads = 16,
            channels=channels,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1)
        
    def forward(self, x1, x2):
        out = self.v(torch.cat([x1,x2],1))
        logits = out[:,:self.dim]
        shift = out[:,-1]

        return logits, shift.squeeze()    


class ViTDiffShiftPredictor(nn.Module):
    def __init__(self,image_size, dim, channels=3, patch_size=32):
        super(ViTDiffShiftPredictor, self).__init__()
        self.dim = dim
        self.v = DeepViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = self.dim+1,
            dim = 1024,
            depth = 6,
            heads = 16,
            channels=channels,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1)
        
    def forward(self, x1, x2):
        x_diff = x2 - x1
        out = self.v(x_diff)
        logits = out[:,:self.dim]
        shift = out[:,-1]

        return logits, shift.squeeze()    
