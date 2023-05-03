#!/usr/bin/env python
# coding: utf-8
# %%
# #!pip install -r requirements_short.txt


# %%
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid

from torch_tools.visualization import to_image
from utils import is_conditional
from visualization import interpolate
from loading import load_trained_from_dir, load_generator

from classifier_networks import VGG, vgg_layers

import numpy as np
from matplotlib import pyplot as plt
import json
import os
import collections


# %%
def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out


# %%
from matplotlib import pyplot as plt

# %%
from glob import glob

# %%
from utils import make_noise, one_hot


# %%
device = torch.device('cuda:0')


# %%
G_weights='./models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt'
root_dir = './results_StyleGAN_5_directions_random' #'./models/pretrained/deformators/StyleGAN2/'
training_name = 'FACE_Attractive' # 'FACE_Attractive_5dirs_random'
classifier_weight_file = 'models/classifiers/celebA_Attractive_vgg11_classifier.pt'

# %%
gan_resolution = 1024
gan_output_channel = 3
shift_in_w = True


# %%
args = json.load(open(os.path.join(root_dir, 'args.json')))
args['w_shift'] = shift_in_w
args['gan_resolution'] = gan_resolution


# %%
G = load_generator(args, G_weights)


# %%
result_dir = root_dir
deformator = load_trained_from_dir(result_dir,G.dim_shift,shift_in_w=shift_in_w)
deformator.eval()


# %%
shift_predictor_lr = 1e-4
n_steps = 100000
batch_size = 1
noise_scale = 0.1
gamma = 0.5


# %%
class_count = 2
classifier_input_size = 256
classifier_input_channel = 3

classifier_weights = torch.load(classifier_weight_file)
if isinstance(classifier_weights, collections.OrderedDict):
    classifier = VGG(vgg_layers,class_count)
    classifier.load_state_dict(classifier_weights)
else:
    classifier = classifier_weights
classifier = classifier.cuda()
classifier = classifier.eval()


# %%
resize_transform = transforms.Resize((classifier_input_size,classifier_input_size))


# %%
class FCShiftPredictor(nn.Module):
    def __init__(self,input_dim,class_dim, inner_dim, output_dim):
        super(FCShiftPredictor, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_


# %%
last_weight = sorted(glob('trained_scale_predictors/shift_model_{}_{:1.3f}_*.pt'.format(training_name, gamma)))[-1]
shift_model = torch.load(last_weight)
shift_model.to(device).eval()

# %%
z = torch.randn([1,G.dim_z]).repeat(batch_size,1).to(device)
z_noise = noise_scale * torch.randn([batch_size,G.dim_z]).to(device)
z_perturb = z + z_noise

z_perturbs = z_perturb.repeat([class_count,1])

target_random = torch.randn([batch_size, class_count]).to(device)
target_classes = torch.argmax(target_random, 1, keepdim=True)
y_target = torch.FloatTensor(batch_size, class_count).to(device)
y_target.zero_()
y_target.scatter_(1, target_classes, 1)

y_targets = torch.eye(class_count,class_count).to(device)

dir_pred = shift_model(z_perturbs,y_targets)

img_shift = G(z_perturb + deformator(dir_pred) )

if gan_resolution != classifier_input_size:
    img_shift = resize_transform(img_shift)

if gan_output_channel == 1 and args.classifier_input_channel > 1:
    img_shift = img_shift.repeat([1,args.classifier_input_channel,1,1])

y_shift = classifier(img_shift)

if isinstance(y_shift,tuple):
    y_shift = y_shift[0]

y_out = torch.softmax(y_shift,1)

scale_loss = torch.mean(torch.abs(dir_pred))


# %%
path_steps = 4
inter_points = torch.cat((linspace(z_perturb + deformator(dir_pred[0]),z_perturb,path_steps),linspace(z_perturb ,z_perturb + deformator(dir_pred[1]),path_steps)[1:]),0)
fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
for step,z_i in enumerate(inter_points):
    img = G(z_i)
    y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
    #print(y)
    img_np = img[0].permute(1,2,0).cpu().detach().numpy()
    img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
    axs[step].imshow((img_norm * 255).astype(np.uint8))
    
    axs[step].set_title(r"C:{:1.2f}".format(y))
    #axs[step].axis('off')
    axs[step].set_xticks([])
    axs[step].set_yticks([])
plt.subplots_adjust(wspace=0, hspace=0)

# %%
top_dirs = torch.argsort(torch.abs(dir_pred[0]),descending=True)

for top_dir in top_dirs[:5]:
    e = torch.FloatTensor(1, deformator.input_dim).to(device)
    e.zero_()
    e.scatter_(1, torch.tensor([[top_dir]]).cuda(),dir_pred[0][top_dir].item())
    
    top_interp_dir = deformator(e)
    inter_points = linspace(z_perturb + top_interp_dir,z_perturb,path_steps)
    
    fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
    for step,z_i in enumerate(inter_points):
        img = G(z_i)
        y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
        #print(y)
        img_np = img[0].permute(1,2,0).cpu().detach().numpy()
        img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
        axs[step].imshow((img_norm * 255).astype(np.uint8))

        axs[step].set_title(r"C:{:1.2f}".format(y))
        #axs[step].axis('off')
        axs[step].set_xticks([])
        axs[step].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)


# %%
c_dir = 1
to_the_right = True

top_dirs = torch.argsort(torch.abs(dir_pred[c_dir]),descending=True)

e = torch.FloatTensor(1, deformator.input_dim).to(device)
e.zero_()
e = e.scatter_(1, torch.tensor(top_dirs[:512].unsqueeze(0)).cuda(),1) * dir_pred[c_dir]

top_interp_dir = deformator(e)
if to_the_right:
    inter_points = linspace(z_perturb, z_perturb + top_interp_dir,path_steps) 
else:
    inter_points = linspace(z_perturb + top_interp_dir,z_perturb,path_steps)

fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
for step,z_i in enumerate(inter_points):
    img = G(z_i)
    y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
    #print(y)
    img_np = img[0].permute(1,2,0).cpu().detach().numpy()
    img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
    axs[step].imshow((img_norm * 255).astype(np.uint8))

    axs[step].set_title(r"C:{:1.2f}".format(y))
    #axs[step].axis('off')
    axs[step].set_xticks([])
    axs[step].set_yticks([])
plt.subplots_adjust(wspace=0, hspace=0)

# %%
c_dir = 1
to_the_right = True

top_dirs = torch.argsort(torch.abs(dir_pred[c_dir]),descending=True)

for i in range(5):
    e = torch.FloatTensor(1, deformator.input_dim).to(device)
    e.zero_()
    e = e.scatter_(1, torch.tensor(top_dirs[i:i+1].unsqueeze(0)).cuda(),1) * dir_pred[c_dir]

    top_interp_dir = deformator(e)
    if to_the_right:
        inter_points = linspace(z_perturb, z_perturb + top_interp_dir,path_steps) 
    else:
        inter_points = linspace(z_perturb + top_interp_dir,z_perturb,path_steps)

    fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
    for step,z_i in enumerate(inter_points):
        img = G(z_i)
        y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
        #print(y)
        img_np = img[0].permute(1,2,0).cpu().detach().numpy()
        img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
        axs[step].imshow((img_norm * 255).astype(np.uint8))

        axs[step].set_title(r"C:{:1.2f}".format(y))
        #axs[step].axis('off')
        axs[step].set_xticks([])
        axs[step].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

# %%
c_dir = 1
to_the_right = True

top_dirs = torch.argsort(torch.abs(dir_pred[c_dir]),descending=True)

z_start = z_perturb
e_res = torch.FloatTensor(1, deformator.input_dim).to(device)
e_res.zero_()

for i in range(5):
    e = torch.FloatTensor(1, deformator.input_dim).to(device)
    e.zero_()
    e = e.scatter_(1, torch.tensor(top_dirs[i:i+1].unsqueeze(0)).cuda(),1) * dir_pred[c_dir]

    top_interp_dir = deformator(e)
    if to_the_right:
        inter_points = linspace(z_start, z_start + top_interp_dir,path_steps) 
    else:
        inter_points = linspace(z_start + top_interp_dir,z_start,path_steps)
    
    z_start = inter_points[-1]

    fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
    for step,z_i in enumerate(inter_points):
        img = G(z_i)
        y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
        #print(y)
        img_np = img[0].permute(1,2,0).cpu().detach().numpy()
        img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
        axs[step].imshow((img_norm * 255).astype(np.uint8))

        axs[step].set_title(r"C:{:1.2f}".format(y))
        #axs[step].axis('off')
        if step == 0:
            axs[step].set_ylabel(top_dirs[i].item())
        axs[step].set_xticks([])
        axs[step].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

# %%
batch_size = 200000
z_perturb = torch.randn([1,G.dim_z]).repeat(batch_size,1).to(device)
y_target = torch.Tensor([[0.0,1.0]]).to(device)
y_target = y_target.repeat([batch_size,1])
dir_pred = shift_model(z_perturb,y_target)
mean_dirs_pred = torch.mean(dir_pred,0)
mean_abs_dir_pred = torch.mean(torch.abs(dir_pred),0)
top_dirs = torch.argsort(mean_abs_dir_pred,descending=True)
#plt.plot(range(512),torch.abs(dir_pred)[0].cpu().detach().numpy())

# %%
torch.save([mean_dirs_pred,mean_abs_dir_pred,top_dirs],'FACE_Attractive_top_causal_interpretable_dirs.pt')

# %%
[mean_dirs_pred,mean_abs_dir_pred,top_dirs] = torch.load('FACE_Attractive_top_causal_interpretable_dirs.pt')

# %%
c_dir = 1
to_the_right = True
path_steps = 4
img_per_dir = 5
z_perturbs = torch.randn([img_per_dir,G.dim_z]).to(device)

for i in range(5):
    e = torch.FloatTensor(1, deformator.input_dim).to(device)
    e.zero_()
    e = e.scatter_(1, torch.tensor(top_dirs[i:i+1].unsqueeze(0)).cuda(),1) * mean_dirs_pred
    top_interp_dir = deformator(e)
    
    for z in z_perturbs:
        z_perturb = z.unsqueeze(0)
        if to_the_right:
            inter_points = linspace(z_perturb, z_perturb + top_interp_dir,path_steps) 
        else:
            inter_points = linspace(z_perturb + top_interp_dir,z_perturb,path_steps)

        fig, axs = plt.subplots(1,len(inter_points),figsize=(15,15))
        for step,z_i in enumerate(inter_points):
            img = G(z_i)
            y = torch.softmax(classifier(resize_transform(img))[0],1)[0][1].item()
            #print(y)
            img_np = img[0].permute(1,2,0).cpu().detach().numpy()
            img_norm = (img_np - img_np.min())/(img_np.max() - img_np.min())
            axs[step].imshow((img_norm * 255).astype(np.uint8))

            axs[step].set_title(r"C:{:1.2f}".format(y))
            if step == 0:
                axs[step].set_ylabel(top_dirs[i].item())
            #axs[step].axis('off')
            axs[step].set_xticks([])
            axs[step].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
